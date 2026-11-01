`timescale 1ns / 1ps
// Prefetch engine with one tracking entry per stream ID (tensor).
//
// MODE 0: off
// MODE 1: next-line  (stride fixed at +1, always on)
// MODE 2: stride     (RPT-style: 2-bit confidence, delta must repeat before issuing)
// MODE 3: TAPS       (tile-aware: a stream is a sequence of rows, each `run` steps
//                     of s1 long, separated by a jump. The row shape and the jump
//                     are learned and trusted separately:
//                       c1 - confidence in s1 / row length -> prefetch to row end
//                       cw - confidence in the jump s2     -> prefetch into next row
//                     so fixed-size blocks at unpredictable places (embedding rows,
//                     KV pages) still get their remaining lines prefetched.)
//
// Training: misses and first touches of prefetched lines always train. A hit on a
// normal (demand) line trains only if it is exactly the address the stream
// expected, so ordinary data reuse does not disturb the learned pattern.
//
// Issue: each stream keeps a frontier (the last address it prefetched) and may run
// up to DEGREE lines ahead of demand. A round-robin arbiter picks one stream per
// cycle and sends the next address on its frontier.
module prefetch_engine #(
    parameter LINE_ADDR_W = 26,
    parameter MODE        = 3,
    parameter DEGREE      = 4,
    parameter NSTREAM     = 8
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   train_valid,
    input  wire [LINE_ADDR_W-1:0] train_line_addr,
    input  wire [2:0]             train_sid,
    input  wire [1:0]             train_kind,
    output wire                   pf_req_valid,
    output wire [LINE_ADDR_W-1:0] pf_req_line_addr,
    input  wire                   pf_req_ready
);
    localparam K_MISS = 2'd0, K_PF_HIT = 2'd1, K_PF_LATE = 2'd2, K_HIT = 2'd3;
    localparam AW = LINE_ADDR_W;
    localparam [2:0] CMAX = (MODE == 2) ? 3'd3 : 3'd7;
    localparam [2:0] TH   = 3'd2;

    reg          e_v     [0:NSTREAM-1];
    reg [AW-1:0] e_last  [0:NSTREAM-1];
    reg [AW-1:0] e_s1    [0:NSTREAM-1];   // step inside a row (two's complement)
    reg [AW-1:0] e_s2    [0:NSTREAM-1];   // jump to the next row
    reg [7:0]    e_run   [0:NSTREAM-1];   // s1 steps per row
    reg          e_rok   [0:NSTREAM-1];   // row length seen twice
    reg          e_wcand [0:NSTREAM-1];   // s2 seen once
    reg          e_wok   [0:NSTREAM-1];   // s2 seen twice
    reg [7:0]    e_pos   [0:NSTREAM-1];   // s1 steps taken in the current row
    reg [2:0]    e_c1    [0:NSTREAM-1];
    reg [2:0]    e_cw    [0:NSTREAM-1];
    reg [3:0]    e_ahead [0:NSTREAM-1];   // lines prefetched beyond demand
    reg [AW-1:0] e_fa    [0:NSTREAM-1];   // frontier address
    reg [7:0]    e_fp    [0:NSTREAM-1];   // frontier row position

    // Next point on a stream's walk from (a, p).
    // Returns {ok, pos, addr}; ok = 0 when the walk cannot be predicted
    // (row ends here and the jump is unknown or untrusted).
    function [AW+8:0] step(input [AW-1:0] a, input [7:0] p, input [2:0] st);
        reg [AW-1:0] s1, s2;
        reg [7:0]    run;
        reg          rok, wok;
        reg [2:0]    c1, cw;
        begin
            s1 = e_s1[st]; s2 = e_s2[st]; run = e_run[st];
            rok = e_rok[st]; wok = e_wok[st]; c1 = e_c1[st]; cw = e_cw[st];
            if (MODE == 3 && rok && p >= run)
                step = {wok && cw >= TH, 8'd0, a + s2};
            else
                step = {(MODE == 3) ? (c1 >= TH) : 1'b1, (p == 8'hFF) ? p : p + 8'd1, a + s1};
        end
    endfunction

    // ------------------------------------------------------- issue arbiter
    reg [2:0]    rr;
    reg          sel_v;
    reg [2:0]    sel, cand;
    reg [AW+8:0] cs, sel_step;
    integer k;

    always @(*) begin
        sel_v = 0; sel = 0; sel_step = 0;
        for (k = NSTREAM - 1; k >= 0; k = k - 1) begin
            cand = rr + k[2:0];
            cs   = step(e_fa[cand], e_fp[cand], cand);
            if (MODE != 0 && e_v[cand] && e_ahead[cand] < DEGREE && e_s1[cand] != 0 &&
                cs[AW+8] && (MODE == 3 || e_c1[cand] >= TH)) begin
                sel_v = 1; sel = cand; sel_step = cs;
            end
        end
    end

    assign pf_req_valid     = sel_v;
    assign pf_req_line_addr = sel_step[AW-1:0];
    wire   issue = sel_v && pf_req_ready;

    // ------------------------------------------------------------ training
    integer i;
    reg [2:0]    t;
    reg [AW-1:0] a, d;
    reg [AW+8:0] ex;
    reg          hitp, consumed, reset_fr;
    reg [7:0]    np;
    reg [2:0]    n1, nw;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rr <= 0;
            for (i = 0; i < NSTREAM; i = i + 1) begin
                e_v[i] <= 0; e_last[i] <= 0; e_s1[i] <= 0; e_s2[i] <= 0; e_run[i] <= 0;
                e_rok[i] <= 0; e_wcand[i] <= 0; e_wok[i] <= 0; e_pos[i] <= 0;
                e_c1[i] <= 0; e_cw[i] <= 0; e_ahead[i] <= 0; e_fa[i] <= 0; e_fp[i] <= 0;
            end
        end else begin
            consumed = 0; reset_fr = 0; t = train_sid; np = 0;
            if (issue) rr <= sel + 3'd1;

            if (train_valid && MODE != 0) begin
                a = train_line_addr;
                if (!e_v[t]) begin
                    e_v[t] <= 1; e_last[t] <= a; e_pos[t] <= 0;
                    e_s1[t] <= (MODE == 1) ? 1 : 0;
                    e_c1[t] <= (MODE == 1) ? 3'd7 : 3'd0;
                    e_cw[t] <= 0; e_run[t] <= 0;
                    e_rok[t] <= 0; e_wcand[t] <= 0; e_wok[t] <= 0;
                    reset_fr = 1;
                end else if (a != e_last[t]) begin
                    d    = a - e_last[t];
                    ex   = step(e_last[t], e_pos[t], t);
                    hitp = (a == ex[AW-1:0]) &&
                           (MODE != 3 || !(e_rok[t] && e_pos[t] >= e_run[t]) || e_wok[t]);

                    if (train_kind != K_HIT || hitp) begin
                        e_last[t] <= a;
                        n1 = e_c1[t]; nw = e_cw[t]; np = e_pos[t];

                        if (MODE == 2) begin
                            if (d == e_s1[t]) begin
                                if (n1 < CMAX) n1 = n1 + 1;
                            end else begin
                                if (n1 <= 1) e_s1[t] <= d;
                                if (n1 != 0) n1 = n1 - 1;
                            end
                        end else if (MODE == 3) begin
                            if (d == e_s1[t] && !(e_rok[t] && e_pos[t] >= e_run[t])) begin
                                // expected step inside the row
                                if (n1 < CMAX) n1 = n1 + 1;
                                np = (e_pos[t] == 8'hFF) ? e_pos[t] : e_pos[t] + 1;
                            end else if (d == e_s1[t]) begin
                                // row is longer than learned
                                e_rok[t] <= 0;
                                n1 = (n1 >= 2) ? n1 - 2 : 0;
                                np = (e_pos[t] == 8'hFF) ? e_pos[t] : e_pos[t] + 1;
                            end else if (e_pos[t] >= 1) begin
                                // a jump at the end of a row
                                if (e_pos[t] == e_run[t]) begin
                                    e_rok[t] <= 1;
                                end else begin
                                    e_run[t] <= e_pos[t];
                                    if (e_rok[t]) n1 = (n1 >= 2) ? n1 - 2 : 0;
                                    e_rok[t] <= 0;
                                end
                                if (d == e_s2[t] && (e_wcand[t] || e_wok[t])) begin
                                    e_wok[t] <= 1;
                                    if (nw < CMAX) nw = nw + 1;
                                end else if (e_wok[t] && nw >= 2) begin
                                    nw = nw - 2;          // keep the pattern, one odd jump
                                end else begin
                                    e_s2[t] <= d; e_wcand[t] <= 1; e_wok[t] <= 0; nw = 0;
                                end
                                np = 0;
                            end else if (n1 >= TH) begin
                                // jump right after a jump, but s1 has been reliable:
                                // assume a new row starts here
                                n1 = n1 - 1; np = 0;
                            end else begin
                                // the step itself changed
                                e_s1[t] <= d; e_rok[t] <= 0; e_run[t] <= 0;
                                n1 = 0; np = 1;
                            end
                        end
                        e_c1[t]  <= n1;
                        e_cw[t]  <= nw;
                        e_pos[t] <= np;

                        if (MODE == 1) hitp = (d == 1);
                        if (MODE == 2) hitp = (d == e_s1[t]);
                        if (hitp && e_ahead[t] != 0) consumed = 1;
                        else                         reset_fr = 1;
                    end
                end
            end

            // frontier bookkeeping
            for (i = 0; i < NSTREAM; i = i + 1) begin
                if (train_valid && MODE != 0 && t == i && reset_fr) begin
                    e_ahead[i] <= 0;
                    e_fa[i]    <= train_line_addr;
                    e_fp[i]    <= np;
                end else begin
                    if (issue && sel == i) begin
                        e_fa[i] <= sel_step[AW-1:0];
                        e_fp[i] <= sel_step[AW+7:AW];
                    end
                    e_ahead[i] <= e_ahead[i] + ((issue && sel == i) ? 4'd1 : 4'd0)
                                             - ((train_valid && t == i && consumed) ? 4'd1 : 4'd0);
                end
            end
        end
    end
endmodule
