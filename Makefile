# Choose what/how you want to run the analyses by changing options in the
# configuration section, then use run-decoding and run-ensemble to train
# models. Update and run the plot target to create a new plot.

# Non-configurable paramters. Don't touch.
USR := $(shell whoami | head -c 2)
NL = $(words $(LAGS))

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD = sbatch --array=1-$(NL) code/run.sh
CMD = echo
CMD = python

# Choose the subject to run for
SID := 676
SID := 625

# Choose model hyper parameters
PARAMS := default
HYPER_PARAMS :=

PARAMS := borgcls
HYPER_PARAMS := --batch-size 608 --lr 0.0019 --dropout 0.11 --reg 0.01269 --reg-head 0.0004 --conv-filters 160 --fine-epochs 300 --patience 120 --half-window 15 --n-weight-avg 30

# Choose which modes to run for: production, comprehension, or both.
MODES := comp
MODES := prod comp
MODES := prod

# Ignore. Choose how many jobs to run for each lag. NOTE - one sbatch job runs multiple
# jobs If sbatch runs 5 in each job, and if LAGX = 2, then you'll get 10 runs
# in total.
LAGX := 1

# Choose the lags to run for (512hz).
# LAGS := $(shell yes "{-1024..1024..256}" | head -n $(LAGX) | tr '\n' ' ')
LAGS = $(shell seq -1024 256 1024)
LAGS = 128

# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

# Ensure that data pickles exist before kicking off any jobs
data-exists:
	@for mode in $(MODES); do \
	    [[ -r data/$(SID)/$(SID)_binned_signal.pkl ]] || echo "[ERROR] $(SID)_binned_signal.pkl does not exist!"; \
	    [[ -r data/$(SID)/$(SID)_$${mode}_labels_MWF30.pkl ]] || echo "[ERROR] $(SID)_$${mode}_labels_MWF30.pkl does not exist!"; \
	done

# General function to run decoding given the configured parameters above.
# Note that run.sh will run an ensemble as well.
run-decoding: data-exists
	for mode in $(MODES); do \
		$(CMD) \
		    code/tfsdec_main.py \
		    --signal-pickle data/$(SID)/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(SID)/$(SID)_$${mode}_labels_MWF30.pkl \
		    --lags $(LAGS) \
		    $(HYPER_PARAMS) \
		    --fine-epochs 1 \
		    --model s_$(SID)-m_$$mode-p_$(PARAMS)-u_$(USR); \
	done

run-ensemble: data-exists
	for mode in $(MODES); do \
		$(CMD) \
		    code/tfsdec_main.py \
		    --signal-pickle data/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(SID)_$${mode}_labels_MWF30.pkl \
		    --lags $(LAGS) \
		    --ensemble \
		    $(HYPER_PARAMS) \
		    --model s_$(SID)-m_$$mode-p_$(PARAMS)-u_$(USR); \
	done

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

plot:
	mkdir -p results/plots/
	python code/plot.py \
	    --q "model == 's_625-m_prod-p_borgcls-u_$(USR)' and ensemble == True" \
	        "model == 's_625-m_comp-p_borgcls-u_$(USR)' and ensemble == True" \
	    --x lag \
	    --y avg_test_rocauc

aggregate-results:
	python code/aggregate_results.py

# -----------------------------------------------------------------------------
#  Misc. targets
# -----------------------------------------------------------------------------

setup:
	mkdir -p /scratch/gpfs/$USER/247-decoding/{data,results,logs}
	ln -s /scratch/gpfs/$USER/247-decoding/data
	ln -s /scratch/gpfs/$USER/247-decoding/logs
	ln -s /scratch/gpfs/$USER/247-decoding/results
	@echo ln -s /scratch/gpfs/$USER/247-pickling/results/* /scratch/gpfs/$USER/247-decoding/data/


# If you have pickled the data yourself, then you can just link to it
link-data:
	echo "revisit this:"
	# ln -sf $(shell dirname `pwd`)/247-pickling/results/* data/

# Otherwise, you can download it from google cloud bucket
download-data:
	echo "revisit this:"
	# gsutil -m rsync gs://247-podcast-data/247_pickles/ data/


sync-plots:
	rsync -aPv \
	    results/plots/ /tigress/$(USER)/247-decoding-results/plots

sync-results:
	rsync -aP --include="*/" --include="*.(png|csv|json)" \
	    results/ /tigress/$(USER)/247-decoding-results

archive-results:
	rsync -aP results/ /tigress/$(USER)/247-decoding-results

print-lags:
	@echo number of lags: $(NL)
	@echo $(LAGS)

