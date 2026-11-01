`timescale 1ns / 1ps

// ============================================================================
// MODULE 1: AI-Aware Prefetch Engine (Watch -> Guess -> Fetch Early Pipeline)
//   PREFETCH_MODE:
//     0 = Disabled (Baseline Cache Only)
//     1 = Sequential Next-Line Prefetcher (Line + 1)
//     2 = Standard Constant-Stride Prefetcher (Line + Delta)
//     3 = TAPS (Proposed AI-Aware 2D Tensor-Wrap + Confidence Throttling)
// ============================================================================
module ai_prefetch_engine #(
    parameter LINE_ADDR_W   = 26,
    parameter PREFETCH_MODE = 3
)(
    input  wire                   clk,
    input  wire                   rst_n,
    // Watch interface (snoops demand accesses from AI workload)
    input  wire                   snoop_valid,
    input  wire [LINE_ADDR_W-1:0] snoop_line_addr,
    // Fetch Early interface (to Cache / DRAM Arbiter)
    output reg                    pf_req_valid,
    output reg  [LINE_ADDR_W-1:0] pf_req_line_addr,
    input  wire                   pf_req_ack
);

    // Stage 1 (WATCH): Track previous line address and compute stride delta
    reg                          watch_valid;
    reg        [LINE_ADDR_W-1:0] prev_line_addr;
    reg        [LINE_ADDR_W-1:0] curr_line_addr;
    reg signed [LINE_ADDR_W-1:0] curr_delta;
    reg                          first_access_done;

    // Stage 2 (GUESS): Predictor registers for Stride & 2D Tensor-Wrap (TAPS)
    reg signed [LINE_ADDR_W-1:0] primary_stride;
    reg signed [LINE_ADDR_W-1:0] wrap_stride;
    reg        [7:0]             cur_run_len;
    reg        [7:0]             learned_run_len;
    reg                          wrap_valid;
    reg        [2:0]             confidence;       // 0..7 saturating counter
    reg        [LINE_ADDR_W-1:0] last_pred_addr;
    reg                          has_last_pred;

    wire signed [LINE_ADDR_W-1:0] raw_delta = $signed(snoop_line_addr) - $signed(prev_line_addr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            watch_valid       <= 1'b0;
            prev_line_addr    <= {LINE_ADDR_W{1'b0}};
            curr_line_addr    <= {LINE_ADDR_W{1'b0}};
            curr_delta        <= {LINE_ADDR_W{1'b0}};
            first_access_done <= 1'b0;

            primary_stride    <= {{(LINE_ADDR_W-1){1'b0}}, 1'b1}; // default +1
            wrap_stride       <= {LINE_ADDR_W{1'b0}};
            cur_run_len       <= 8'd0;
            learned_run_len   <= 8'd0;
            wrap_valid        <= 1'b0;
            confidence        <= 3'd4; // start mid-high so sequential starts smoothly
            last_pred_addr    <= {LINE_ADDR_W{1'b0}};
            has_last_pred     <= 1'b0;

            pf_req_valid      <= 1'b0;
            pf_req_line_addr  <= {LINE_ADDR_W{1'b0}};
        end else begin
            // Clear pending prefetch request when acknowledged by cache controller
            if (pf_req_ack) begin
                pf_req_valid <= 1'b0;
            end

            // ----------------------------------------------------------------
            // STAGE 1: WATCH - Snoop AI Core memory accesses & extract delta
            // ----------------------------------------------------------------
            if (snoop_valid) begin
                curr_line_addr    <= snoop_line_addr;
                prev_line_addr    <= snoop_line_addr;
                first_access_done <= 1'b1;
                if (first_access_done && (snoop_line_addr != prev_line_addr)) begin
                    curr_delta  <= raw_delta;
                    watch_valid <= 1'b1;
                end else begin
                    watch_valid <= 1'b0;
                end
            end else begin
                watch_valid <= 1'b0;
            end

            // ----------------------------------------------------------------
            // STAGE 2 & 3: GUESS & FETCH EARLY - Update AI Predictor & Issue
            // ----------------------------------------------------------------
            if (watch_valid) begin
                case (PREFETCH_MODE)
                    // Mode 0: Prefetching Disabled
                    0: begin
                        pf_req_valid <= 1'b0;
                    end

                    // Mode 1: Next-Line Sequential Prefetcher
                    1: begin
                        pf_req_valid     <= 1'b1;
                        pf_req_line_addr <= curr_line_addr + 26'd1;
                    end

                    // Mode 2: Standard Constant-Stride Prefetcher
                    2: begin
                        primary_stride   <= curr_delta;
                        pf_req_valid     <= 1'b1;
                        pf_req_line_addr <= $signed(curr_line_addr) + curr_delta;
                    end

                    // Mode 3: TAPS (2D Nested-Stride Tensor Engine + Confidence Filter)
                    default: begin
                        // 1. Update Confidence based on whether previous guess matched
                        if (has_last_pred) begin
                            if (curr_line_addr == last_pred_addr) begin
                                if (confidence < 3'd7) confidence <= confidence + 3'd1;
                            end else begin
                                if (confidence >= 3'd2) confidence <= confidence - 3'd2;
                                else                    confidence <= 3'd0;
                            end
                        end

                        // 2. 2D Tensor Pattern State Machine
                        if (curr_delta == primary_stride) begin
                            // Continuing along intra-row stride (delta_1)
                            if (wrap_valid && (cur_run_len >= learned_run_len)) begin
                                // Stream exceeded 2D tile width -> adapt back to 1D stream
                                wrap_valid  <= 1'b0;
                                cur_run_len <= cur_run_len + 8'd1;
                                pf_req_line_addr <= $signed(curr_line_addr) + primary_stride;
                                last_pred_addr   <= $signed(curr_line_addr) + primary_stride;
                            end else if (wrap_valid && ((cur_run_len + 8'd1) == learned_run_len)) begin
                                // Reached exact end of 2D tensor row -> Predict Row-Wrap Jump (delta_2)!
                                cur_run_len      <= cur_run_len + 8'd1;
                                pf_req_line_addr <= $signed(curr_line_addr) + wrap_stride;
                                last_pred_addr   <= $signed(curr_line_addr) + wrap_stride;
                            end else begin
                                // Normal intra-row prediction (delta_1)
                                cur_run_len      <= cur_run_len + 8'd1;
                                pf_req_line_addr <= $signed(curr_line_addr) + primary_stride;
                                last_pred_addr   <= $signed(curr_line_addr) + primary_stride;
                            end
                            has_last_pred <= 1'b1;
                            pf_req_valid  <= (confidence >= 3'd2);

                        end else if (wrap_valid && (curr_delta == wrap_stride) && (cur_run_len > 8'd0)) begin
                            // Confirmed 2D row-wrap transition! Reset column counter & predict +primary_stride
                            learned_run_len  <= cur_run_len;
                            cur_run_len      <= 8'd0;
                            if (confidence < 3'd7) confidence <= confidence + 3'd1;
                            pf_req_line_addr <= $signed(curr_line_addr) + primary_stride;
                            last_pred_addr   <= $signed(curr_line_addr) + primary_stride;
                            has_last_pred    <= 1'b1;
                            pf_req_valid     <= 1'b1;

                        end else if (cur_run_len >= 8'd2) begin
                            // Just completed a run of >=2 primary_stride steps and saw a jump:
                            // Record this jump as a candidate 2D row-wrap stride!
                            wrap_stride      <= curr_delta;
                            learned_run_len  <= cur_run_len;
                            wrap_valid       <= 1'b1;
                            cur_run_len      <= 8'd0;
                            pf_req_line_addr <= $signed(curr_line_addr) + primary_stride;
                            last_pred_addr   <= $signed(curr_line_addr) + primary_stride;
                            has_last_pred    <= 1'b1;
                            pf_req_valid     <= (confidence >= 3'd2);

                        end else begin
                            // Irregular jump or new 1D stride transition
                            primary_stride   <= curr_delta;
                            cur_run_len      <= 8'd1;
                            wrap_valid       <= 1'b0;
                            pf_req_line_addr <= $signed(curr_line_addr) + curr_delta;
                            last_pred_addr   <= $signed(curr_line_addr) + curr_delta;
                            has_last_pred    <= 1'b1;
                            // Throttle prefetching if confidence drops on chaotic Recommender accesses
                            pf_req_valid     <= (confidence >= 3'd3);
                        end
                    end
                endcase
            end
        end
    end
endmodule


// ============================================================================
// MODULE 2: 4-Way Set-Associative Cache Controller with Tree-PLRU &
//           Anti-Pollution Prefetch Tagging
// ============================================================================
module ai_cache_controller #(
    parameter LINE_ADDR_W   = 26,
    parameter SET_BITS      = 4,   // 16 Sets x 4 Ways = 64 Lines (4 KB Cache)
    parameter WAYS          = 4,
    parameter DRAM_LATENCY  = 16,  // 16 cycles for DRAM demand miss / fill
    parameter ANTI_POLLUTE  = 1    // 1 = Protect demand lines from unconfirmed prefetches
)(
    input  wire                   clk,
    input  wire                   rst_n,
    // AI Workload Demand Interface
    input  wire                   req_valid,
    input  wire [LINE_ADDR_W-1:0] req_line_addr,
    output reg                    resp_ready,
    output reg                    resp_hit,
    // Prefetch Engine Interface
    input  wire                   pf_req_valid,
    input  wire [LINE_ADDR_W-1:0] pf_req_line_addr,
    output reg                    pf_req_ack,
    // Snoop Output to Prefetch Engine
    output reg                    snoop_valid,
    output reg  [LINE_ADDR_W-1:0] snoop_line_addr,
    // Hardware Performance Counters (for Level 5 Validation)
    output reg  [31:0]            cnt_accesses,
    output reg  [31:0]            cnt_hits,
    output reg  [31:0]            cnt_misses,
    output reg  [31:0]            cnt_pf_issued,
    output reg  [31:0]            cnt_pf_useful,
    output reg  [31:0]            cnt_total_Read_cycles
);

    localparam NUM_SETS = (1 << SET_BITS);
    localparam TAG_W    = LINE_ADDR_W - SET_BITS;

    // Tag Array, Valid bits, Prefetch-Unconfirmed bits (pf_bit), and 3-bit Tree-PLRU
    reg [TAG_W-1:0] tag_array   [0:NUM_SETS-1][0:WAYS-1];
    reg             valid_array [0:NUM_SETS-1][0:WAYS-1];
    reg             pf_bit      [0:NUM_SETS-1][0:WAYS-1];
    reg [2:0]       plru_bits   [0:NUM_SETS-1]; // [2]=right subtree, [1]=left subtree, [0]=root

    // Controller FSM States
    localparam ST_IDLE        = 2'd0;
    localparam ST_DEMAND_MISS = 2'd1;
    localparam ST_PF_FILL     = 2'd2;

    reg [1:0]             state;
    reg [4:0]             dram_timer;
    reg [LINE_ADDR_W-1:0] active_demand_addr;
    reg [LINE_ADDR_W-1:0] active_pf_addr;
    reg                   pending_demand;

    // Decode helper signals for demand request
    wire [SET_BITS-1:0] req_set = req_line_addr[SET_BITS-1:0];
    wire [TAG_W-1:0]    req_tag = req_line_addr[LINE_ADDR_W-1:SET_BITS];

    // Check hit across 4 ways for demand request
    reg        hit_found;
    reg [1:0]  hit_way;
    integer    w, s;

    always @(*) begin
        hit_found = 1'b0;
        hit_way   = 2'd0;
        for (w = 0; w < WAYS; w = w + 1) begin
            if (valid_array[req_set][w] && (tag_array[req_set][w] == req_tag)) begin
                hit_found = 1'b1;
                hit_way   = w[1:0];
            end
        end
    end

    // Check if prefetch candidate is ALREADY in cache (Cache Filter)
    wire [SET_BITS-1:0] pf_set = pf_req_line_addr[SET_BITS-1:0];
    wire [TAG_W-1:0]    pf_tag = pf_req_line_addr[LINE_ADDR_W-1:SET_BITS];
    reg                 pf_already_cached;

    always @(*) begin
        pf_already_cached = 1'b0;
        for (w = 0; w < WAYS; w = w + 1) begin
            if (valid_array[pf_set][w] && (tag_array[pf_set][w] == pf_tag)) begin
                pf_already_cached = 1'b1;
            end
        end
    end

    // Victim Selection Function (Invalid Way -> Unconfirmed Prefetch Way -> Tree-PLRU Way)
    function [1:0] select_victim(input [SET_BITS-1:0] idx);
        reg [2:0] b;
        begin
            b = plru_bits[idx];
            if (!valid_array[idx][0])      select_victim = 2'd0;
            else if (!valid_array[idx][1]) select_victim = 2'd1;
            else if (!valid_array[idx][2]) select_victim = 2'd2;
            else if (!valid_array[idx][3]) select_victim = 2'd3;
            else if (ANTI_POLLUTE && pf_bit[idx][0]) select_victim = 2'd0;
            else if (ANTI_POLLUTE && pf_bit[idx][1]) select_victim = 2'd1;
            else if (ANTI_POLLUTE && pf_bit[idx][2]) select_victim = 2'd2;
            else if (ANTI_POLLUTE && pf_bit[idx][3]) select_victim = 2'd3;
            else begin
                // Standard 3-bit Tree-PLRU victim lookup
                if (b[0] == 1'b0)
                    select_victim = (b[1] == 1'b0) ? 2'd0 : 2'd1;
                else
                    select_victim = (b[2] == 1'b0) ? 2'd2 : 2'd3;
            end
        end
    endfunction

    // Update Tree-PLRU bits to point AWAY from recently accessed way
    task update_plru(input [SET_BITS-1:0] idx, input [1:0] accessed_way);
        begin
            case (accessed_way)
                2'd0: begin plru_bits[idx][0] <= 1'b1; plru_bits[idx][1] <= 1'b1; end
                2'd1: begin plru_bits[idx][0] <= 1'b1; plru_bits[idx][1] <= 1'b0; end
                2'd2: begin plru_bits[idx][0] <= 1'b0; plru_bits[idx][2] <= 1'b1; end
                2'd3: begin plru_bits[idx][0] <= 1'b0; plru_bits[idx][2] <= 1'b0; end
            endcase
        end
    endtask

    reg [1:0] victim_w;
    wire [SET_BITS-1:0] miss_set = active_demand_addr[SET_BITS-1:0];
    wire [TAG_W-1:0]    miss_tag = active_demand_addr[LINE_ADDR_W-1:SET_BITS];
    wire [SET_BITS-1:0] fill_pf_set = active_pf_addr[SET_BITS-1:0];
    wire [TAG_W-1:0]    fill_pf_tag = active_pf_addr[LINE_ADDR_W-1:SET_BITS];

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state                 <= ST_IDLE;
            dram_timer            <= 5'd0;
            resp_ready            <= 1'b0;
            resp_hit              <= 1'b0;
            pf_req_ack            <= 1'b0;
            snoop_valid           <= 1'b0;
            snoop_line_addr       <= {LINE_ADDR_W{1'b0}};
            active_demand_addr    <= {LINE_ADDR_W{1'b0}};
            active_pf_addr        <= {LINE_ADDR_W{1'b0}};
            pending_demand        <= 1'b0;

            cnt_accesses          <= 32'd0;
            cnt_hits              <= 32'd0;
            cnt_misses            <= 32'd0;
            cnt_pf_issued         <= 32'd0;
            cnt_pf_useful         <= 32'd0;
            cnt_total_Read_cycles <= 32'd0;

            for (s = 0; s < NUM_SETS; s = s + 1) begin
                plru_bits[s] <= 3'b000;
                for (w = 0; w < WAYS; w = w + 1) begin
                    valid_array[s][w] <= 1'b0;
                    tag_array[s][w]   <= {TAG_W{1'b0}};
                    pf_bit[s][w]      <= 1'b0;
                end
            end
        end else begin
            resp_ready  <= 1'b0;
            pf_req_ack  <= 1'b0;
            snoop_valid <= 1'b0;

            // Count cycles whenever a demand request is active/waiting
            if (req_valid || pending_demand || (state == ST_DEMAND_MISS)) begin
                cnt_total_Read_cycles <= cnt_total_Read_cycles + 32'd1;
            end

            case (state)
                // ------------------------------------------------------------
                // IDLE: Serve Demand Request in 1 cycle on Hit, or start DRAM
                // ------------------------------------------------------------
                ST_IDLE: begin
                    if (req_valid || pending_demand) begin
                        pending_demand  <= 1'b0;
                        cnt_accesses    <= cnt_accesses + 32'd1;
                        snoop_valid     <= 1'b1;
                        snoop_line_addr <= req_line_addr;

                        if (hit_found) begin
                            // 1-Cycle Cache Hit!
                            resp_ready <= 1'b1;
                            resp_hit   <= 1'b1;
                            cnt_hits   <= cnt_hits + 32'd1;
                            update_plru(req_set, hit_way);

                            // If this line was brought in by the Prefetch Engine,
                            // record a USEFUL PREFETCH and promote to confirmed demand line!
                            if (pf_bit[req_set][hit_way]) begin
                                cnt_pf_useful           <= cnt_pf_useful + 32'd1;
                                pf_bit[req_set][hit_way] <= 1'b0;
                            end
                        end else begin
                            // Demand Cache Miss -> Fetch from DRAM
                            active_demand_addr <= req_line_addr;
                            cnt_misses         <= cnt_misses + 32'd1;
                            dram_timer         <= DRAM_LATENCY - 1;
                            state              <= ST_DEMAND_MISS;
                        end
                    end else if (pf_req_valid) begin
                        // Bus is idle: evaluate background Prefetch Request
                        pf_req_ack <= 1'b1;
                        if (!pf_already_cached) begin
                            active_pf_addr <= pf_req_line_addr;
                            cnt_pf_issued  <= cnt_pf_issued + 32'd1;
                            dram_timer     <= (DRAM_LATENCY / 2); // Pipelined burst prefetch
                            state          <= ST_PF_FILL;
                        end
                    end
                end

                // ------------------------------------------------------------
                // DEMAND MISS: Wait for DRAM, install line, return to AI Core
                // ------------------------------------------------------------
                ST_DEMAND_MISS: begin
                    if (dram_timer == 5'd0) begin
                        victim_w = select_victim(miss_set);
                        valid_array[miss_set][victim_w] <= 1'b1;
                        tag_array[miss_set][victim_w]   <= miss_tag;
                        pf_bit[miss_set][victim_w]      <= 1'b0;
                        update_plru(miss_set, victim_w);

                        resp_ready <= 1'b1;
                        resp_hit   <= 1'b0;
                        state      <= ST_IDLE;
                    end else begin
                        dram_timer <= dram_timer - 5'd1;
                    end
                end

                // ------------------------------------------------------------
                // BACKGROUND PREFETCH FILL: Overlap with AI compute cycles
                // ------------------------------------------------------------
                ST_PF_FILL: begin
                    // Remember if AI core issues a demand request during background fill
                    if (req_valid) begin
                        pending_demand <= 1'b1;
                    end

                    // If a demand request targets the EXACT line currently being prefetched:
                    if ((req_valid || pending_demand) && (req_line_addr == active_pf_addr)) begin
                        if (dram_timer == 5'd0) begin
                            victim_w = select_victim(fill_pf_set);
                            valid_array[fill_pf_set][victim_w] <= 1'b1;
                            tag_array[fill_pf_set][victim_w]   <= fill_pf_tag;
                            pf_bit[fill_pf_set][victim_w]      <= 1'b0; // Immediately used!
                            update_plru(fill_pf_set, victim_w);

                            pending_demand  <= 1'b0;
                            cnt_accesses    <= cnt_accesses + 32'd1;
                            cnt_hits        <= cnt_hits + 32'd1;
                            cnt_pf_useful   <= cnt_pf_useful + 32'd1;
                            snoop_valid     <= 1'b1;
                            snoop_line_addr <= req_line_addr;
                            resp_ready      <= 1'b1;
                            resp_hit        <= 1'b1;
                            state           <= ST_IDLE;
                        end else begin
                            dram_timer <= dram_timer - 5'd1;
                        end
                    end else if (dram_timer == 5'd0) begin
                        // Complete background prefetch installation
                        victim_w = select_victim(fill_pf_set);
                        valid_array[fill_pf_set][victim_w] <= 1'b1;
                        tag_array[fill_pf_set][victim_w]   <= fill_pf_tag;
                        pf_bit[fill_pf_set][victim_w]      <= 1'b1; // Mark as speculative prefetch
                        // Note: If ANTI_POLLUTE=1, we do NOT update PLRU on unconfirmed prefetch
                        // so it remains the first eviction candidate if the guess was wrong!
                        if (!ANTI_POLLUTE) begin
                            update_plru(fill_pf_set, victim_w);
                        end
                        state <= ST_IDLE;
                    end else begin
                        dram_timer <= dram_timer - 5'd1;
                    end
                end
            endcase
        end
    end
endmodule


// ============================================================================
// MODULE 3: Top-Level AI Memory Subsystem (Cache + Prefetch Engine)
// ============================================================================
module ai_memory_subsystem #(
    parameter LINE_ADDR_W   = 26,
    parameter SET_BITS      = 4,
    parameter WAYS          = 4,
    parameter DRAM_LATENCY  = 16,
    parameter PREFETCH_MODE = 3
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        req_valid,
    input  wire [31:0] req_byte_addr,
    output wire        resp_ready,
    output wire        resp_hit,
    output wire [31:0] cnt_accesses,
    output wire [31:0] cnt_hits,
    output wire [31:0] cnt_misses,
    output wire [31:0] cnt_pf_issued,
    output wire [31:0] cnt_pf_useful,
    output wire [31:0] cnt_total_Read_cycles
);

    wire [LINE_ADDR_W-1:0] req_line_addr = req_byte_addr[31:6]; // 64-byte cache line
    wire                   snoop_valid;
    wire [LINE_ADDR_W-1:0] snoop_line_addr;
    wire                   pf_req_valid;
    wire [LINE_ADDR_W-1:0] pf_req_line_addr;
    wire                   pf_req_ack;

    ai_cache_controller #(
        .LINE_ADDR_W  (LINE_ADDR_W),
        .SET_BITS     (SET_BITS),
        .WAYS         (WAYS),
        .DRAM_LATENCY (DRAM_LATENCY),
        .ANTI_POLLUTE (PREFETCH_MODE == 3 ? 1 : 0)
    ) u_cache (
        .clk                   (clk),
        .rst_n                 (rst_n),
        .req_valid             (req_valid),
        .req_line_addr         (req_line_addr),
        .resp_ready            (resp_ready),
        .resp_hit              (resp_hit),
        .pf_req_valid          (pf_req_valid),
        .pf_req_line_addr      (pf_req_line_addr),
        .pf_req_ack            (pf_req_ack),
        .snoop_valid           (snoop_valid),
        .snoop_line_addr       (snoop_line_addr),
        .cnt_accesses          (cnt_accesses),
        .cnt_hits              (cnt_hits),
        .cnt_misses            (cnt_misses),
        .cnt_pf_issued         (cnt_pf_issued),
        .cnt_pf_useful         (cnt_pf_useful),
        .cnt_total_Read_cycles (cnt_total_Read_cycles)
    );

    ai_prefetch_engine #(
        .LINE_ADDR_W   (LINE_ADDR_W),
        .PREFETCH_MODE (PREFETCH_MODE)
    ) u_prefetcher (
        .clk              (clk),
        .rst_n            (rst_n),
        .snoop_valid      (snoop_valid),
        .snoop_line_addr  (snoop_line_addr),
        .pf_req_valid     (pf_req_valid),
        .pf_req_line_addr (pf_req_line_addr),
        .pf_req_ack       (pf_req_ack)
    );

endmodule