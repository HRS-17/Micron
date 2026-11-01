`timescale 1ns / 1ps
// Cache + prefetch engine. 64-byte lines.
module mem_subsystem #(
    parameter LINE_ADDR_W    = 26,
    parameter SET_BITS       = 6,
    parameter DRAM_LATENCY   = 32,
    parameter DRAM_ISSUE_GAP = 2,
    parameter PF_MSHRS       = 8,
    parameter PF_MODE        = 3,
    parameter PF_DEGREE      = 4,
    parameter ANTI_POLLUTE   = 0
)(
    input  wire        clk,
    input  wire        rst_n,
    input  wire        req_valid,
    input  wire [31:0] req_byte_addr,
    input  wire [2:0]  req_sid,
    output wire        resp_ready,
    output wire        resp_hit,
    output wire [31:0] cnt_accesses,
    output wire [31:0] cnt_hits,
    output wire [31:0] cnt_misses,
    output wire [31:0] cnt_late,
    output wire [31:0] cnt_pf_issued,
    output wire [31:0] cnt_pf_dropped,
    output wire [31:0] cnt_pf_useful,
    output wire [31:0] cnt_pf_useless,
    output wire [31:0] cnt_dram_reads
);
    wire [LINE_ADDR_W-1:0] line = req_byte_addr[31:6];

    wire                   tr_v;
    wire [LINE_ADDR_W-1:0] tr_a;
    wire [2:0]             tr_s;
    wire [1:0]             tr_k;
    wire                   pf_v, pf_rdy;
    wire [LINE_ADDR_W-1:0] pf_a;

    cache_ctrl #(
        .LINE_ADDR_W(LINE_ADDR_W), .SET_BITS(SET_BITS), .DRAM_LATENCY(DRAM_LATENCY),
        .DRAM_ISSUE_GAP(DRAM_ISSUE_GAP), .PF_MSHRS(PF_MSHRS), .ANTI_POLLUTE(ANTI_POLLUTE)
    ) u_cache (
        .clk(clk), .rst_n(rst_n),
        .req_valid(req_valid), .req_line_addr(line), .req_sid(req_sid),
        .resp_ready(resp_ready), .resp_hit(resp_hit),
        .pf_req_valid(pf_v), .pf_req_line_addr(pf_a), .pf_req_ready(pf_rdy),
        .train_valid(tr_v), .train_line_addr(tr_a), .train_sid(tr_s), .train_kind(tr_k),
        .cnt_accesses(cnt_accesses), .cnt_hits(cnt_hits), .cnt_misses(cnt_misses),
        .cnt_late(cnt_late), .cnt_pf_issued(cnt_pf_issued), .cnt_pf_dropped(cnt_pf_dropped),
        .cnt_pf_useful(cnt_pf_useful), .cnt_pf_useless(cnt_pf_useless),
        .cnt_dram_reads(cnt_dram_reads)
    );

    prefetch_engine #(
        .LINE_ADDR_W(LINE_ADDR_W), .MODE(PF_MODE), .DEGREE(PF_DEGREE)
    ) u_pf (
        .clk(clk), .rst_n(rst_n),
        .train_valid(tr_v), .train_line_addr(tr_a), .train_sid(tr_s), .train_kind(tr_k),
        .pf_req_valid(pf_v), .pf_req_line_addr(pf_a), .pf_req_ready(pf_rdy)
    );
endmodule
