# Choose what/how you want to run the analyses by changing options in the
# configuration section, then use run-decoding and run-ensemble to train
# models. Update and run the plot target to create a new plot.

# Non-configurable paramters. Don't touch.
USR := $(shell whoami | head -c 2)
NL = $(words $(LAGS))

# -----------------------------------------------------------------------------
#  Configurable options
# -----------------------------------------------------------------------------

PJCT := podcast
PJCT := tfs


# Model options
# ---------------

# Regression or classification
MODN := classify
MODE := --classify

MODN := regress
MODE := 

# Choose model hyper parameters
PARAMS := default
HYPER_PARAMS :=

PARAMS := borgcls
HYPER_PARAMS := --batch-size 608 --lr 0.0019 --dropout 0.11 --reg 0.01269 --reg-head 0.0004 --conv-filters 160 --epochs 300 --patience 120 --half-window 312.5 --n-weight-avg 30

PARAMS := vsr
HYPER_PARAMS := --batch-size 256 --lr 0.00025 --dropout 0.21 --reg 0.003 --reg-head 0.0005 --conv-filters 160 --epochs 1500 --patience 150 --half-window 312.5 --n-weight-avg 20

# Dataset options
# ---------------

# Choose the subject to run for
SID := 625
BC := 

# SID := 676
# BC := --bad-convos 38 39

# SID := 7170
# # BC := --bad-convos 2 23 24
# BC :=

# SIG_FN := --sig-elec-file data/717_21-conv-elec-189.csv
# SIG_FN := --sig-elec-file data/tfs-sig-file-glove-625-comp.csv
SIG_FN := 

# SID := 777
# SIG_FN := --sig-elec-file data/129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file data/164-phase-5000-sig-elec-gpt2xl50d-perElec-FDR-01-LH.csv
# SIG_FN := --sig-elec-file data/160-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH_newVer.csv

NE = 160

# Choose embedddings
# glove
EMBN := glove50
CNTX := 1
LAYER_IDX := 1
PCA :=

# gpt2
EMBN = gpt2-xl
CNTX = 1024
LAYER_IDX = 48
PCA := --pca 50


# Align with others
ALIGN_WITH = --align-with gpt2-xl
ALIGN_WITH = 

# Minimum word frequency
MWF := 0

# Choose which modes to run for: production, comprehension, or both.
MODES := prod
MODES := prod comp
MODES := comp

# Running options
# ---------------

# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM all lags in parallel.
CMD = echo
CMD = python
CMD = sbatch --array=1-$(NL) scripts/run.sh

# misc flags
MISC := --epochs 1
MISC :=

# Ignore. Choose how many jobs to run for each lag. NOTE - one sbatch job runs multiple
# jobs If sbatch runs 5 in each job, and if LAGX = 2, then you'll get 10 runs
# in total.
LAGX := 1

# Choose the lags to run for in ms
# LAGS := $(shell yes "{-1024..1024..256}" | head -n $(LAGX) | tr '\n' ' ')
LAGS = 0
LAGS = -2000 $(shell seq -1000 250 2000)

# Datum Modification
DM := shiftn1
DM := all
DM := shift

# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

# General function to run decoding given the configured parameters above.
# Note that run.sh will run an ensemble as well.
run-decoding:
	for mode in $(MODES); do \
	    $(CMD) scripts/tfsdec_main.py \
	        --signal-pickle data/$(PJCT)/$(SID)/pickles/$(SID)_binned_signal.pkl \
	        --label-dir data/$(PJCT)/$(SID)/pickles/embeddings/$(EMBN)/full \
	        --lags $(LAGS) \
			$(BC) \
	        $(HYPER_PARAMS) \
	        --mode $${mode} \
		--min-dev-freq $(MWF) --min-test-freq $(MWF) \
		--verbose 0 \
		$(SIG_FN) \
		$(ALIGN_WITH) \
		--datum-mod $(DM) \
	    --context-len $(CNTX) \
	    --layer-idx $(LAYER_IDX) \
	        $(PCA) \
	        $(MODE) \
	        $(MISC) \
	        --model decoding-s-$(SID)_e-$(NE)_t-$(MODN)_m-$${mode}_e-$(EMBN)_p-$(PARAMS)_mwf-$(MWF)-sig-$(DM); \
	done

# In case you need to run the ensemble on its own
run-ensemble:
	for mode in $(MODES); do \
		$(CMD) \
		    scripts/tfsdec_main.py \
		    --signal-pickle data/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(SID)_$${mode}_labels_MWF30.pkl \
		    --lags $(LAGS) \
		    --ensemble \
		    $(HYPER_PARAMS) \
		    --model s-$(SID)_t-$(MODN)_m-$${mode}_e-$(EMBN)_p-$(PARAMS); \
	done

# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

plots: aggregate-results plot sync-plots

	    # --q "model == 's-777_m-comp_e-glove50_p-borgcls' and ensemble == True and lag >= -512 and lag <= 512" \
	    #     "model == 's-777_m-comp_e-gpt2-xl_p-borgcls' and ensemble == True and lag >= -512 and lag <= 512"
	    #     "model == 's-777_t-regress_m-comp_e-blenderbot-small_p-borgcls' and ensemble == True and lag >= -512 and lag <= 512"
	    #     "model == 's-777_t-regress_m-comp_e-glove50_p-borgcls' and ensemble == True and lag >= -512 and lag <= 512" \
	    #     "model == 's-777_t-regress_m-comp_e-gpt2-xl_p-borgcls' and ensemble == True and lag >= -512 and lag <= 512" \
	    #     "model == 's-777_t-regress_m-comp_e-gpt2-xl_p-borgcls_mwf-5' and ensemble == True and lag >= -512 and lag <= 512" \
	    #     "model == 's-777_e-164_t-regress_m-comp_e-gpt2-xl_p-borgcls_mwf-5' and ensemble == True and lag >= -512 and lag <= 512" \
	    #     "model == 's-777_e-164_t-regress_m-comp_e-gpt2-xl_p-vsr_mwf-5' and ensemble == True and lag >= -512 and lag <= 512" \

plot:
	rm -f results/plots/*
	mkdir -p results/plots/
	python scripts/plot.py \
	    --q "model == 'decoding-s-625_e-160_t-regress_m-prod_e-glove50_p-vsr_mwf-0-sig-all' and ensemble == True" \
	    	"model == 'decoding-s-625_e-160_t-regress_m-comp_e-glove50_p-vsr_mwf-0-sig-all' and ensemble == True" \
	    	"model == 'decoding-s-625_e-160_t-regress_m-prod_e-gpt2-xl_p-vsr_mwf-0-sig-shift' and ensemble == True" \
	    	"model == 'decoding-s-625_e-160_t-regress_m-comp_e-gpt2-xl_p-vsr_mwf-0-sig-shift' and ensemble == True" \
	    --labels prod-glove comp-glove prod-gpt2 comp-gpt2  \
	    --x lag \
	    --y avg_test_nn_rocauc_test_w_avg \
	    --output results/plots/625-new
	rsync -av results/plots/ ~/tigress/247-decoding-results/

aggregate-results:
	python scripts/aggregate_results.py
	cp -f results/aggregate.csv /tigress/kw1166/247-decoding-results/


# -----------------------------------------------------------------------------
#  Misc. targets
# -----------------------------------------------------------------------------

setup:
	mkdir -p /scratch/gpfs/$(USER)/247-decoding/{data,results,logs}
	# ln -s /scratch/gpfs/$(USER)/247-decoding/data
	# ln -s /scratch/gpfs/$(USER)/247-decoding/logs
	# ln -s /scratch/gpfs/$(USER)/247-decoding/results
	ln -s /scratch/gpfs/$(USER)/247-pickling/results/* /scratch/gpfs/$(USER)/247-decoding/data/


# If you have pickled the data yourself, then you can just link to it
PDIR := $(shell dirname `pwd`)
link-data:
	ln -fs $(PDIR)/247-pickling/results/* data/



# link-data:
# 	echo "revisit this:"
# 	# ln -sf $(shell dirname `pwd`)/247-pickling/results/* data/

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
