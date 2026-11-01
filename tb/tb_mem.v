`timescale 1ns / 1ps
// Runs one trace through six configurations in parallel and appends one CSV row each.
//   plusargs: +trace=<file> +name=<label> +csv=<file> [+dump]
//   parameters (set with iverilog -P): DRAM_LAT, GAP, DEGREE, ISSUE_GAP
//
//   cfg 0: no prefetch          cfg 3: stride + anti-pollution
//   cfg 1: next-line            cfg 4: TAPS
//   cfg 2: stride               cfg 5: TAPS + anti-pollution
module tb_mem;
    parameter DRAM_LAT  = 32;
    parameter GAP       = 8;     // compute cycles between memory requests
    parameter DEGREE    = 4;
    parameter ISSUE_GAP = 2;
    parameter MAXLEN    = 32768;
    localparam NCFG     = 6;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0;
    integer cyc = 0;
    always @(posedge clk) cyc <= cyc + 1;

    reg [39:0] mem [0:MAXLEN-1];
    integer n;

    reg        start = 0;
    reg        req_valid [0:NCFG-1];
    reg [31:0] req_addr  [0:NCFG-1];
    reg [2:0]  req_sid   [0:NCFG-1];
    wire       resp_ready[0:NCFG-1];
    wire       resp_hit  [0:NCFG-1];
    wire [31:0] c_acc[0:NCFG-1], c_hit[0:NCFG-1], c_miss[0:NCFG-1], c_late[0:NCFG-1];
    wire [31:0] c_iss[0:NCFG-1], c_drop[0:NCFG-1], c_use[0:NCFG-1], c_usel[0:NCFG-1], c_dram[0:NCFG-1];
    reg        done     [0:NCFG-1];
    integer    lat_sum  [0:NCFG-1];
    integer    end_cyc  [0:NCFG-1];

    localparam [17:0] MODES = {3'd3, 3'd3, 3'd2, 3'd2, 3'd1, 3'd0};   // cfg 5..0
    localparam [5:0]  APOL  = 6'b101000;

    genvar g;
    generate
        for (g = 0; g < NCFG; g = g + 1) begin : cfg
            mem_subsystem #(
                .DRAM_LATENCY(DRAM_LAT), .DRAM_ISSUE_GAP(ISSUE_GAP), .PF_DEGREE(DEGREE),
                .PF_MODE(MODES[3*g +: 3]), .ANTI_POLLUTE(APOL[g])
            ) dut (
                .clk(clk), .rst_n(rst_n),
                .req_valid(req_valid[g]), .req_byte_addr(req_addr[g]), .req_sid(req_sid[g]),
                .resp_ready(resp_ready[g]), .resp_hit(resp_hit[g]),
                .cnt_accesses(c_acc[g]), .cnt_hits(c_hit[g]), .cnt_misses(c_miss[g]),
                .cnt_late(c_late[g]), .cnt_pf_issued(c_iss[g]), .cnt_pf_dropped(c_drop[g]),
                .cnt_pf_useful(c_use[g]), .cnt_pf_useless(c_usel[g]), .cnt_dram_reads(c_dram[g])
            );

            integer idx, t0, k;
            initial begin
                req_valid[g] = 0; req_addr[g] = 0; req_sid[g] = 0;
                done[g] = 0; lat_sum[g] = 0;
                wait (start);
                for (idx = 0; idx < n; idx = idx + 1) begin
                    @(posedge clk);
                    req_valid[g] <= 1;
                    req_addr[g]  <= mem[idx][31:0];
                    req_sid[g]   <= mem[idx][34:32];
                    t0 = cyc;
                    @(posedge clk);
                    req_valid[g] <= 0;
                    while (!resp_ready[g]) @(posedge clk);
                    lat_sum[g] = lat_sum[g] + (cyc - t0);
                    for (k = 0; k < GAP; k = k + 1) @(posedge clk);
                end
                end_cyc[g] = cyc;
                done[g] = 1;
            end
        end
    endgenerate

    reg [8*256-1:0] trace_file, name, csv_file;
    integer fd, i, all_done;
    real hr, al, acc;

    initial begin
        if (!$value$plusargs("trace=%s", trace_file)) trace_file = "traces/trace_llm.hex";
        if (!$value$plusargs("name=%s", name))        name = "llm";
        if (!$value$plusargs("csv=%s", csv_file))     csv_file = "results/rtl_results.csv";
        if ($test$plusargs("dump")) begin
            $dumpfile("waveform.vcd");
            $dumpvars(0, tb_mem);
        end

        for (i = 0; i < MAXLEN; i = i + 1) mem[i] = 40'hFF_FFFF_FFFF;
        $readmemh(trace_file, mem);
        n = 0;
        while (n < MAXLEN && mem[n] != 40'hFF_FFFF_FFFF) n = n + 1;

        repeat (4) @(posedge clk);
        rst_n = 1;
        repeat (2) @(posedge clk);
        start = 1;

        all_done = 0;
        while (!all_done) begin
            @(posedge clk);
            all_done = 1;
            for (i = 0; i < NCFG; i = i + 1) if (!done[i]) all_done = 0;
        end
        repeat (2) @(posedge clk);

        fd = $fopen(csv_file, "a");
        for (i = 0; i < NCFG; i = i + 1) begin
            if (c_acc[i] != n) $display("ERROR: cfg %0d counted %0d accesses, trace has %0d", i, c_acc[i], n);
            hr  = 100.0 * c_hit[i] / c_acc[i];
            al  = 1.0 * lat_sum[i] / n;
            acc = (c_iss[i] > 0) ? 100.0 * c_use[i] / c_iss[i] : 0.0;
            $fdisplay(fd, "%0s,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%.3f,%.3f,%0d,%0d,%0d,%0d,%0d,%.2f,%0d",
                name, i, DRAM_LAT, GAP, DEGREE, ISSUE_GAP, n, c_hit[i], c_miss[i], c_late[i],
                hr, al, end_cyc[i], c_iss[i], c_drop[i], c_use[i], c_usel[i], acc, c_dram[i]);
            $display("%0s cfg%0d  hit %6.2f%%  miss %5d  late %5d  lat %6.2f  cycles %7d  pf %5d  acc %6.2f%%  useless %5d  dram %5d",
                name, i, hr, c_miss[i], c_late[i], al, end_cyc[i], c_iss[i], acc, c_usel[i], c_dram[i]);
        end
        $fclose(fd);
        $finish;
    end
endmodule
