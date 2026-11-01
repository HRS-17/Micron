.PHONY: all quick plots wave clean

all:            ## traces, RTL runs, golden check, sweeps, figures
	python3 scripts/run_all.py
	python3 scripts/plots.py

quick:          ## traces, main RTL run, golden check
	python3 scripts/run_all.py --quick

plots:
	python3 scripts/plots.py

wave:           ## VCD of the recommender trace (smallest)
	mkdir -p build
	iverilog -g2012 -o build/sim_wave tb/tb_mem.v rtl/*.v
	vvp -n build/sim_wave +trace=traces/trace_rec.hex +name=rec +csv=build/wave.csv +dump

clean:
	rm -rf build waveform.vcd
