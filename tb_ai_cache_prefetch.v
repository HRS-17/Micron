`timescale 1ns / 1ps

module tb_ai_cache_prefetch;

    localparam TRACE_LEN   = 256;
    localparam COMPUTE_GAP = 12; // 12 AI MAC compute cycles between memory requests

    reg clk;
    reg rst_n;

    initial clk = 1'b0;
    always #5 clk = ~clk; // 100 MHz clock

    // Trace memory arrays
    reg [31:0] trace_llm [0:TRACE_LEN-1];
    reg [31:0] trace_cnn [0:TRACE_LEN-1];
    reg [31:0] trace_rec [0:TRACE_LEN-1];
    reg [31:0] active_trace [0:TRACE_LEN-1];

    // Per-DUT signals (0=Baseline, 1=Sequential, 2=Stride, 3=TAPS AI-Aware)
    reg         req_valid [0:3];
    reg  [31:0] req_addr  [0:3];
    wire        resp_ready[0:3];
    wire        resp_hit  [0:3];

    wire [31:0] cnt_accesses [0:3];
    wire [31:0] cnt_hits     [0:3];
    wire [31:0] cnt_misses   [0:3];
    wire [31:0] cnt_pf_issued[0:3];
    wire [31:0] cnt_pf_useful[0:3];
    wire [31:0] cnt_cycles   [0:3];

    genvar m;
    generate
        for (m = 0; m < 4; m = m + 1) begin : gen_dut
            ai_memory_subsystem #(
                .PREFETCH_MODE(m)
            ) u_dut (
                .clk                   (clk),
                .rst_n                 (rst_n),
                .req_valid             (req_valid[m]),
                .req_byte_addr         (req_addr[m]),
                .resp_ready            (resp_ready[m]),
                .resp_hit              (resp_hit[m]),
                .cnt_accesses          (cnt_accesses[m]),
                .cnt_hits              (cnt_hits[m]),
                .cnt_misses            (cnt_misses[m]),
                .cnt_pf_issued         (cnt_pf_issued[m]),
                .cnt_pf_useful         (cnt_pf_useful[m]),
                .cnt_total_Read_cycles (cnt_cycles[m])
            );
        end
    endgenerate

    integer csv_fd;
    integer i, gap;

    // Task to drive a single DUT through active_trace[0..TRACE_LEN-1]
    task run_dut_zero;
        integer idx, g;
        begin
            for (idx = 0; idx < TRACE_LEN; idx = idx + 1) begin
                @(posedge clk);
                req_valid[0] <= 1'b1;
                req_addr[0]  <= active_trace[idx];
                @(posedge clk);
                req_valid[0] <= 1'b0;
                while (!resp_ready[0]) @(posedge clk);
                for (g = 0; g < COMPUTE_GAP; g = g + 1) @(posedge clk);
            end
        end
    endtask

    task run_dut_one;
        integer idx, g;
        begin
            for (idx = 0; idx < TRACE_LEN; idx = idx + 1) begin
                @(posedge clk);
                req_valid[1] <= 1'b1;
                req_addr[1]  <= active_trace[idx];
                @(posedge clk);
                req_valid[1] <= 1'b0;
                while (!resp_ready[1]) @(posedge clk);
                for (g = 0; g < COMPUTE_GAP; g = g + 1) @(posedge clk);
            end
        end
    endtask

    task run_dut_two;
        integer idx, g;
        begin
            for (idx = 0; idx < TRACE_LEN; idx = idx + 1) begin
                @(posedge clk);
                req_valid[2] <= 1'b1;
                req_addr[2]  <= active_trace[idx];
                @(posedge clk);
                req_valid[2] <= 1'b0;
                while (!resp_ready[2]) @(posedge clk);
                for (g = 0; g < COMPUTE_GAP; g = g + 1) @(posedge clk);
            end
        end
    endtask

    task run_dut_three;
        integer idx, g;
        begin
            for (idx = 0; idx < TRACE_LEN; idx = idx + 1) begin
                @(posedge clk);
                req_valid[3] <= 1'b1;
                req_addr[3]  <= active_trace[idx];
                @(posedge clk);
                req_valid[3] <= 1'b0;
                while (!resp_ready[3]) @(posedge clk);
                for (g = 0; g < COMPUTE_GAP; g = g + 1) @(posedge clk);
            end
        end
    endtask

    task run_all_modes_and_report(input [8*16-1:0] workload_name);
        integer k;
        real hit_rate, amat, pf_acc;
        begin
            rst_n = 1'b0;
            for (k = 0; k < 4; k = k + 1) begin
                req_valid[k] = 1'b0;
                req_addr[k]  = 32'd0;
            end
            repeat (5) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);

            fork
                run_dut_zero;
                run_dut_one;
                run_dut_two;
                run_dut_three;
            join

            $display("\n=====================================================================================");
            $display(" WORKLOAD: %s (256 Memory Requests, 4-Way Tree-PLRU Cache)", workload_name);
            $display("=====================================================================================");
            $display(" Mode                  | Hits  | Misses | Hit Rate | Avg Latency | PF Issued | PF Accuracy");
            $display("-----------------------+-------+--------+----------+-------------+-----------+------------");

            for (k = 0; k < 4; k = k + 1) begin
                hit_rate = (cnt_hits[k] * 100.0) / cnt_accesses[k];
                amat     = (cnt_cycles[k] * 1.0) / cnt_accesses[k];
                if (cnt_pf_issued[k] > 0)
                    pf_acc = (cnt_pf_useful[k] * 100.0) / cnt_pf_issued[k];
                else
                    pf_acc = 0.0;

                case (k)
                    0: $display(" 0: No Prefetch        | %5d |  %5d |  %6.2f%% |  %6.2f cyc |     %5d |     %6.2f%%",
                                cnt_hits[k], cnt_misses[k], hit_rate, amat, cnt_pf_issued[k], pf_acc);
                    1: $display(" 1: Sequential (+1)    | %5d |  %5d |  %6.2f%% |  %6.2f cyc |     %5d |     %6.2f%%",
                                cnt_hits[k], cnt_misses[k], hit_rate, amat, cnt_pf_issued[k], pf_acc);
                    2: $display(" 2: Standard Stride    | %5d |  %5d |  %6.2f%% |  %6.2f cyc |     %5d |     %6.2f%%",
                                cnt_hits[k], cnt_misses[k], hit_rate, amat, cnt_pf_issued[k], pf_acc);
                    3: $display(" 3: TAPS AI-Aware(Ours)| %5d |  %5d |  %6.2f%% |  %6.2f cyc |     %5d |     %6.2f%%",
                                cnt_hits[k], cnt_misses[k], hit_rate, amat, cnt_pf_issued[k], pf_acc);
                endcase

                $fdisplay(csv_fd, "%0s,%0d,%0d,%0d,%.2f,%.2f,%0d,%0d,%.2f",
                          workload_name, k, cnt_hits[k], cnt_misses[k],
                          hit_rate, amat, cnt_pf_issued[k], cnt_pf_useful[k], pf_acc);
            end
        end
    endtask

    initial begin
        $dumpfile("waveform_ai_cache.vcd");
        $dumpvars(0, tb_ai_cache_prefetch);

        $readmemh("trace_llm.hex", trace_llm);
        $readmemh("trace_cnn.hex", trace_cnn);
        $readmemh("trace_rec.hex", trace_rec);

        csv_fd = $fopen("rtl_results.csv", "w");
        $fdisplay(csv_fd, "workload,mode,hits,misses,hit_rate,avg_latency,pf_issued,pf_useful,pf_accuracy");

        // 1. Run LLM Workload
        for (i = 0; i < TRACE_LEN; i = i + 1) active_trace[i] = trace_llm[i];
        run_all_modes_and_report("LLM_Inference   ");

        // 2. Run CNN 2D Tensor Workload
        for (i = 0; i < TRACE_LEN; i = i + 1) active_trace[i] = trace_cnn[i];
        run_all_modes_and_report("CNN_2D_Tensor   ");

        // 3. Run Recommender Workload
        for (i = 0; i < TRACE_LEN; i = i + 1) active_trace[i] = trace_rec[i];
        run_all_modes_and_report("Recommender_Sys ");

        $fclose(csv_fd);
        $display("\n[OK] Saved hardware simulation metrics -> rtl_results.csv");
        $display("[OK] Saved VCD waveform dump         -> waveform_ai_cache.vcd\n");
        $finish;
    end

endmodule