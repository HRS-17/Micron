`timescale 1ns / 1ps
// 4-way set-associative cache (tags only), tree-PLRU, non-blocking prefetch fills.
//
// Timing model
//   hit                   : response 1 cycle after the request
//   demand miss           : waits for a free DRAM issue slot, then DRAM_LATENCY cycles
//   prefetch              : same DRAM_LATENCY, tracked in PF_MSHRS miss registers
//   DRAM issue rate       : one new request every DRAM_ISSUE_GAP cycles (demand first)
//   install port          : one line installed per cycle (demand first)
//
// Every demand access is reported on the train_* port with its kind, so the
// prefetcher can decide what to learn from.
module cache_ctrl #(
    parameter LINE_ADDR_W    = 26,
    parameter SET_BITS       = 6,    // 64 sets x 4 ways x 64 B = 16 KB
    parameter DRAM_LATENCY   = 32,
    parameter DRAM_ISSUE_GAP = 2,
    parameter PF_MSHRS       = 8,
    parameter ANTI_POLLUTE   = 0     // evict unused prefetched lines first
)(
    input  wire                   clk,
    input  wire                   rst_n,
    // demand port (req_valid is a one-cycle pulse, one request outstanding)
    input  wire                   req_valid,
    input  wire [LINE_ADDR_W-1:0] req_line_addr,
    input  wire [2:0]             req_sid,
    output reg                    resp_ready,
    output reg                    resp_hit,
    // prefetch request port
    input  wire                   pf_req_valid,
    input  wire [LINE_ADDR_W-1:0] pf_req_line_addr,
    output wire                   pf_req_ready,
    // training port to the prefetcher
    output reg                    train_valid,
    output reg  [LINE_ADDR_W-1:0] train_line_addr,
    output reg  [2:0]             train_sid,
    output reg  [1:0]             train_kind,
    // counters
    output reg  [31:0]            cnt_accesses,
    output reg  [31:0]            cnt_hits,       // includes first hits on prefetched lines
    output reg  [31:0]            cnt_misses,     // full demand misses
    output reg  [31:0]            cnt_late,       // demand found its line still in flight
    output reg  [31:0]            cnt_pf_issued,  // prefetches sent to DRAM
    output reg  [31:0]            cnt_pf_dropped, // prefetches already cached / in flight
    output reg  [31:0]            cnt_pf_useful,  // prefetched lines later used (timely + late)
    output reg  [31:0]            cnt_pf_useless, // prefetched lines evicted unused
    output reg  [31:0]            cnt_dram_reads
);
    localparam K_MISS = 2'd0, K_PF_HIT = 2'd1, K_PF_LATE = 2'd2, K_HIT = 2'd3;
    localparam NUM_SETS = 1 << SET_BITS;
    localparam TAG_W    = LINE_ADDR_W - SET_BITS;
    localparam TW       = 8;   // timer width

    reg [TAG_W-1:0] tag_q   [0:NUM_SETS-1][0:3];
    reg             valid_q [0:NUM_SETS-1][0:3];
    reg             pf_q    [0:NUM_SETS-1][0:3];
    reg [2:0]       plru_q  [0:NUM_SETS-1];

    // demand miss state
    reg                   dem_issue_pend;   // waiting for a DRAM slot
    reg                   dem_inflight;
    reg [TW-1:0]          dem_timer;
    reg [LINE_ADDR_W-1:0] dem_addr;
    reg                   dem_wait_mshr;    // waiting on a late prefetch

    // prefetch miss registers
    reg                   m_v      [0:PF_MSHRS-1];
    reg                   m_dem    [0:PF_MSHRS-1];   // a demand is waiting on it
    reg [TW-1:0]          m_timer  [0:PF_MSHRS-1];
    reg [LINE_ADDR_W-1:0] m_addr   [0:PF_MSHRS-1];

    reg [TW-1:0] gap_cnt;

    // ---------------------------------------------------------------- helpers
    function [SET_BITS-1:0] set_of(input [LINE_ADDR_W-1:0] a);
        set_of = a[SET_BITS-1:0];
    endfunction
    function [TAG_W-1:0] tag_of(input [LINE_ADDR_W-1:0] a);
        tag_of = a[LINE_ADDR_W-1:SET_BITS];
    endfunction

    // returns {found, way}
    function [2:0] lookup(input [LINE_ADDR_W-1:0] a);
        integer w;
        begin
            lookup = 3'b000;
            for (w = 0; w < 4; w = w + 1)
                if (valid_q[set_of(a)][w] && tag_q[set_of(a)][w] == tag_of(a))
                    lookup = {1'b1, w[1:0]};
        end
    endfunction

    // returns {found, index}
    function [4:0] mshr_match(input [LINE_ADDR_W-1:0] a);
        integer i;
        begin
            mshr_match = 5'd0;
            for (i = PF_MSHRS - 1; i >= 0; i = i - 1)
                if (m_v[i] && m_addr[i] == a) mshr_match = {1'b1, i[3:0]};
        end
    endfunction

    function [1:0] victim(input [SET_BITS-1:0] s);
        reg [2:0] b;
        begin
            b = plru_q[s];
            if      (!valid_q[s][0]) victim = 2'd0;
            else if (!valid_q[s][1]) victim = 2'd1;
            else if (!valid_q[s][2]) victim = 2'd2;
            else if (!valid_q[s][3]) victim = 2'd3;
            else if (ANTI_POLLUTE && pf_q[s][0]) victim = 2'd0;
            else if (ANTI_POLLUTE && pf_q[s][1]) victim = 2'd1;
            else if (ANTI_POLLUTE && pf_q[s][2]) victim = 2'd2;
            else if (ANTI_POLLUTE && pf_q[s][3]) victim = 2'd3;
            else if (b[0] == 1'b0) victim = b[1] ? 2'd1 : 2'd0;
            else                   victim = b[2] ? 2'd3 : 2'd2;
        end
    endfunction

    // point the tree away from the touched way
    function [2:0] plru_touch(input [2:0] b, input [1:0] w);
        case (w)
            2'd0: plru_touch = {b[2], 1'b1, 1'b1};
            2'd1: plru_touch = {b[2], 1'b0, 1'b1};
            2'd2: plru_touch = {1'b1, b[1], 1'b0};
            default: plru_touch = {1'b0, b[1], 1'b0};
        endcase
    endfunction

    // ------------------------------------------------- prefetch accept logic
    reg [PF_MSHRS-1:0] m_v_vec;
    integer vi;
    always @(*) for (vi = 0; vi < PF_MSHRS; vi = vi + 1) m_v_vec[vi] = m_v[vi];

    wire mshr_free = ~&m_v_vec;
    // demand misses get the DRAM slot first
    assign pf_req_ready = mshr_free && (gap_cnt == 0) && !dem_issue_pend;

    // ------------------------------------------------------------ main logic
    integer s, w, i;
    reg [2:0]             lk;
    reg [4:0]             mm;
    reg                   inst_do, inst_dem, inst_used;
    reg [LINE_ADDR_W-1:0] inst_addr;
    reg [3:0]             inst_idx;
    reg [1:0]             vw;
    reg                   gap_taken;
    reg [3:0]             free_idx;
    reg                   free_found;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            resp_ready <= 0; resp_hit <= 0;
            train_valid <= 0; train_line_addr <= 0; train_sid <= 0; train_kind <= 0;
            cnt_accesses <= 0; cnt_hits <= 0; cnt_misses <= 0; cnt_late <= 0;
            cnt_pf_issued <= 0; cnt_pf_dropped <= 0; cnt_pf_useful <= 0;
            cnt_pf_useless <= 0; cnt_dram_reads <= 0;
            dem_issue_pend <= 0; dem_inflight <= 0; dem_timer <= 0; dem_addr <= 0;
            dem_wait_mshr <= 0; gap_cnt <= 0;
            for (i = 0; i < PF_MSHRS; i = i + 1) begin
                m_v[i] <= 0; m_dem[i] <= 0; m_timer[i] <= 0; m_addr[i] <= 0;
            end
            for (s = 0; s < NUM_SETS; s = s + 1) begin
                plru_q[s] <= 3'b000;
                for (w = 0; w < 4; w = w + 1) begin
                    valid_q[s][w] <= 0; tag_q[s][w] <= 0; pf_q[s][w] <= 0;
                end
            end
        end else begin
            resp_ready  <= 0;
            train_valid <= 0;
            gap_taken    = 0;
            if (gap_cnt != 0) gap_cnt <= gap_cnt - 1;

            // ---- 1. pick one line to install this cycle (demand first)
            inst_do = 0; inst_dem = 0; inst_used = 0; inst_idx = 0; inst_addr = 0;
            if (dem_inflight && dem_timer == 0) begin
                inst_do = 1; inst_dem = 1; inst_used = 1; inst_addr = dem_addr;
            end else begin
                for (i = PF_MSHRS - 1; i >= 0; i = i - 1)
                    if (m_v[i] && m_timer[i] == 0) begin
                        inst_do = 1; inst_idx = i[3:0]; inst_addr = m_addr[i];
                        inst_used = m_dem[i];
                    end
            end

            // ---- 2. demand lookup
            if (req_valid) begin
                cnt_accesses    <= cnt_accesses + 1;
                train_valid     <= 1;
                train_line_addr <= req_line_addr;
                train_sid       <= req_sid;
                lk = lookup(req_line_addr);
                mm = mshr_match(req_line_addr);
                if (lk[2]) begin
                    resp_ready <= 1; resp_hit <= 1;
                    cnt_hits   <= cnt_hits + 1;
                    plru_q[set_of(req_line_addr)] <= plru_touch(plru_q[set_of(req_line_addr)], lk[1:0]);
                    if (pf_q[set_of(req_line_addr)][lk[1:0]]) begin
                        pf_q[set_of(req_line_addr)][lk[1:0]] <= 0;
                        cnt_pf_useful <= cnt_pf_useful + 1;
                        train_kind    <= K_PF_HIT;
                    end else begin
                        train_kind    <= K_HIT;
                    end
                end else if (inst_do && !inst_dem && inst_addr == req_line_addr) begin
                    // prefetched line lands this very cycle
                    resp_ready <= 1; resp_hit <= 1;
                    cnt_hits      <= cnt_hits + 1;
                    cnt_pf_useful <= cnt_pf_useful + 1;
                    train_kind    <= K_PF_HIT;
                    inst_used      = 1;
                end else if (mm[4]) begin
                    m_dem[mm[3:0]] <= 1;
                    dem_wait_mshr  <= 1;
                    cnt_late       <= cnt_late + 1;
                    cnt_pf_useful  <= cnt_pf_useful + 1;
                    train_kind     <= K_PF_LATE;
                end else begin
                    cnt_misses     <= cnt_misses + 1;
                    dem_addr       <= req_line_addr;
                    dem_issue_pend <= 1;
                    train_kind     <= K_MISS;
                end
            end

            // ---- 3. install
            if (inst_do) begin
                vw = victim(set_of(inst_addr));
                if (valid_q[set_of(inst_addr)][vw] && pf_q[set_of(inst_addr)][vw])
                    cnt_pf_useless <= cnt_pf_useless + 1;
                valid_q[set_of(inst_addr)][vw] <= 1;
                tag_q  [set_of(inst_addr)][vw] <= tag_of(inst_addr);
                pf_q   [set_of(inst_addr)][vw] <= !inst_used;
                // an unused prefetch does not touch PLRU when anti-pollution is on
                if (inst_used || !ANTI_POLLUTE)
                    plru_q[set_of(inst_addr)] <= plru_touch(plru_q[set_of(inst_addr)], vw);
                if (inst_dem) begin
                    dem_inflight <= 0;
                    resp_ready <= 1; resp_hit <= 0;
                end else begin
                    m_v[inst_idx] <= 0; m_dem[inst_idx] <= 0;
                    if (m_dem[inst_idx]) begin
                        dem_wait_mshr <= 0;
                        resp_ready <= 1; resp_hit <= 0;
                    end
                end
            end

            // ---- 4. timers
            if (dem_inflight && dem_timer != 0) dem_timer <= dem_timer - 1;
            for (i = 0; i < PF_MSHRS; i = i + 1)
                if (m_v[i] && m_timer[i] != 0) m_timer[i] <= m_timer[i] - 1;

            // ---- 5. issue the pending demand miss to DRAM
            if (dem_issue_pend && gap_cnt == 0) begin
                dem_issue_pend <= 0;
                dem_inflight   <= 1;
                dem_timer      <= DRAM_LATENCY - 1;
                cnt_dram_reads <= cnt_dram_reads + 1;
                gap_cnt        <= DRAM_ISSUE_GAP - 1;
                gap_taken       = 1;
            end

            // ---- 6. accept a prefetch request
            if (pf_req_valid && pf_req_ready && !gap_taken) begin
                lk = lookup(pf_req_line_addr);
                mm = mshr_match(pf_req_line_addr);
                if (lk[2] || mm[4] ||
                    ((dem_issue_pend || dem_inflight) && dem_addr == pf_req_line_addr) ||
                    (req_valid && req_line_addr == pf_req_line_addr)) begin
                    cnt_pf_dropped <= cnt_pf_dropped + 1;
                end else begin
                    free_found = 0; free_idx = 0;
                    for (i = PF_MSHRS - 1; i >= 0; i = i - 1)
                        if (!m_v[i] && !(inst_do && !inst_dem && inst_idx == i)) begin
                            free_found = 1; free_idx = i[3:0];
                        end
                    if (free_found) begin
                        m_v[free_idx]     <= 1;
                        m_dem[free_idx]   <= 0;
                        m_addr[free_idx]  <= pf_req_line_addr;
                        m_timer[free_idx] <= DRAM_LATENCY - 1;
                        cnt_pf_issued     <= cnt_pf_issued + 1;
                        cnt_dram_reads    <= cnt_dram_reads + 1;
                        gap_cnt           <= DRAM_ISSUE_GAP - 1;
                    end
                end
            end
        end
    end
endmodule
