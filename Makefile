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
CMD = echo
CMD = python
CMD = sbatch --array=1-$(NL) code/run.sh

# Choose the subject to run for
SIG_FN :=
SID := 676
SID := 625

SID := 777
SIG_FN := --sig-elec-file data/129-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH.csv
SIG_FN := --sig-elec-file data/164-phase-5000-sig-elec-gpt2xl50d-perElec-FDR-01-LH.csv
SIG_FN := --sig-elec-file data/160-phase-5000-sig-elec-glove50d-perElec-FDR-01-LH_newVer.csv

# specify the number of electrodese to be part of the output directory
NE = 160

# Choose model hyper parameters
PARAMS := default
HYPER_PARAMS :=

PARAMS := borgcls
HYPER_PARAMS := --batch-size 608 --lr 0.0019 --dropout 0.11 --reg 0.01269 --reg-head 0.0004 --conv-filters 160 --epochs 300 --patience 120 --half-window 312.5 --n-weight-avg 30

PARAMS := vsr
HYPER_PARAMS := --batch-size 256 --lr 0.00025 --dropout 0.21 --reg 0.003 --reg-head 0.0005 --conv-filters 160 --epochs 1500 --patience 150 --half-window 312.5 --n-weight-avg 20


MWF := 5

# Choose which modes to run for: production, comprehension, or both.
MODES := prod comp
MODES := prod
MODES := comp

# Regression or classification
MODE := --classify
MODN := classify

MODE := 
MODN := regress

# preprocessing options
PCA := --pca 50

# Choose embedddings
EMB := blenderbot-small

EMB := $(SID)_full_glove50_layer_01_embeddings.pkl
EMBN = glove50
PCA :=

EMB := $(SID)_full_gpt2-xl_cnxt_1024_layer_48_embeddings.pkl
EMBN = gpt2xl

ALIGN_WITH = --align-with gpt2 blenderbot_small_90M
ALIGN_WITH = --align-with gpt2

# misc changes
MISC := --epochs 1
MISC :=


# Ignore. Choose how many jobs to run for each lag. NOTE - one sbatch job runs multiple
# jobs If sbatch runs 5 in each job, and if LAGX = 2, then you'll get 10 runs
# in total.
LAGX := 1

# Choose the lags to run for (512hz).
# 16 is 1s, 8 is 0.5s, 4 is 0.25s
# LAGS := $(shell yes "{-1024..1024..256}" | head -n $(LAGX) | tr '\n' ' ')
LAGS = $(shell seq -512 64 512)
LAGS = 250

PJCT := tfs
PJCT := podcast

# -----------------------------------------------------------------------------
# Decoding
# -----------------------------------------------------------------------------

# General function to run decoding given the configured parameters above.
# Note that run.sh will run an ensemble as well.
	        #--signal-pickle data/$(PJCT)/$(SID)/pickles/$(SID)_full_signal.pkl \
		--label-pickle data/756e-fold0.pickle 
run-decoding:
	for mode in $(MODES); do \
	    $(CMD) code/tfsdec_main.py \
	        --signal-pickle data/$(PJCT)/$(SID)/pickles/$(SID)_binned_signal.pkl \
	        --label-pickle data/$(PJCT)/$(SID)/pickles/$(EMB) \
	        --lags $(LAGS) \
	        $(HYPER_PARAMS) \
	        --mode $${mode} \
		--min-dev-freq $(MWF) --min-test-freq $(MWF) \
		--verbose 0 \
		$(SIG_FN) \
		$(ALIGN_WITH) \
	        $(PCA) \
	        $(MODE) \
	        $(EMBP) \
	        $(MISC) \
	        --model latest-s-$(SID)_e-$(NE)_t-$(MODN)_m-$${mode}_e-$(EMBN)_p-$(PARAMS)_mwf-$(MWF); \
	done

run-decoding-single:
	for lag in {1..$(NL)}; do \
	    for mode in $(MODES); do \
		python code/tfsdec_main.py \
		    --lag $${lag} \
		    --signal-pickle data/$(PJCT)/$(SID)/pickles/$(SID)_binned_signal.pkl \
		    --label-pickle data/$(PJCT)/$(SID)/pickles/$(SID)_full_$(EMB)_cnxt_1024_embeddings.pkl \
		    --lags $(LAGS) \
		    $(HYPER_PARAMS) \
		    --mode $${mode} \
		    $(PCA) \
		    $(MODE) \
		    $(EMBP) \
		    $(MISC) \
		    --model s-$(SID)_t-$(MODN)_m-$${mode}_e-$(EMBN)_p-$(PARAMS); \
	    done; \
	done

run-ensemble:
	for mode in $(MODES); do \
		$(CMD) \
		    code/tfsdec_main.py \
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
	mkdir -p results/plots/
	python code/plot.py \
	    --q "model == 's-777_e-160_t-regress_m-comp_e-glove50_p-vsr_mwf-5' and ensemble == True and lag >= -512 and lag <= 512" \
	        "model == 's-777_e-160_t-regress_m-comp_e-gpt2-xl_p-vsr_mwf-5' and ensemble == True and lag >= -512 and lag <= 512" \
	    --labels glove gpt2 \
	    --x lag \
	    --y avg_test_nn_rocauc_test_w_avg \
	    --output results/plots/s-777_e-160_t-regress_m-comp_e-gg_p-borgcls

aggregate-results:
	python code/aggregate_results.py
	cp -f results/aggregate.csv /tigress/zzada/247-decoding-results/


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

