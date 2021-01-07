# Choose what/how you want to run the analyses by changing options in the
# configuration section, then use run-decoding and run-ensemble to train
# models. Update and run the plot target to create a new plot.


# Non-configurable paramters. Don't touch.
USR := $(shell whoami | head -c 2)
NL = $(words $(LAGS))

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

# Choose the command to run:python runs locally, echo is for debugging, sbatch
# is for actual running.
CMD = echo
CMD = python
CMD = bash code/run.sh
CMD = sbatch --array=1-$(NL) code/run.sh

# Choose the subject to run for
SID := 676
SID := 625

# Choose which modes to run for: production, comprehension, or both.
MODES := prod comp
MODES := prod
MODES := comp

# Choose how many jobs to run for each lag. NOTE - one sbatch job runs multiple
# jobs If sbatch runs 5 in each job, and if LAGX = 2, then you'll get 10 runs
# in total.
LAGX := 1

# Choose the lags to run for.
LAGS := $(shell yes "{-2048..2048..256}" | head -n $(LAGX) | tr '\n' ' ')
LAGS := -2048 -1024 -512 0 512 1024 2048
LAGS := 0

# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

# Ensure that data pickles exist before kicking off any jobs
data-exists:
	@for mode in $(MODES); do \
	    [[ -r data/$(SID)_binned_signal.pkl ]] || echo "[ERROR] $(SID)_binned_signal.pkl does not exist!"; \
	    [[ -r data/$(SID)_$${mode}_labels_MWF30.pkl ]] || echo "[ERROR] $(SID)_$${mode}_labels_MWF30.pkl does not exist!"; \
	done


run-decoding: data-exists
	for mode in $(MODES); do \
		$(CMD) 5 \
		    code/tfsdec_main.py \
		    --signal-pickle data/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(SID)_$${mode}_labels_MWF30.pkl \
		    --lags $(LAGS) \
		    --model s_$(SID)-m_$$mode-e_64-u_$(USR); \
	done

run-ensemble: data-exists
	for mode in $(MODES); do \
		$(CMD) 1 \
		    code/tfsdec_main.py \
		    --signal-pickle data/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(SID)_prod_labels_MWF30.pkl \
		    --lags $(LAGS) \
		    --ensemble \
		    --model s_$(SID)-m_prod-e_64-u_$(USR); \
	done

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

plot:
	python code/plot.py \
	    -q "model == 's_625-m_prod-e_64-u_zz' and ensemble == True" \
	       "model == 's_625-m_comp-e_64-u_zz' and ensemble == True" \
	    -x lag \
	    -y avg_rocauc_test_w_avg
	rsync -azp results/plots ~/tigress/

# -----------------------------------------------------------------------------
#  Debugging targets
# -----------------------------------------------------------------------------

print-lags:
	@echo number of lags: $(NL)
	@echo $(LAGS)

