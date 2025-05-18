# Set up directories
REPLICATION_DIR := scripts/replications
MARKER_DIR := .make_markers

# Find all directories in scripts/replications
REPLICATION_DIRS := $(shell find $(REPLICATION_DIR) -maxdepth 1 -type d)
# Extract just the folder names for targeting
REPLICATIONS := $(notdir $(filter-out $(REPLICATION_DIR),$(REPLICATION_DIRS)))
# Generate marker file paths
MARKERS := $(addprefix $(MARKER_DIR)/,$(REPLICATIONS))

# Create directory for markers
$(shell mkdir -p $(MARKER_DIR))

# Default target to run all replications
.PHONY: all_replications
all_replications: $(MARKERS)

# Wildcard rule for running replications
# This will match any replication where both the directory and script exist
$(MARKER_DIR)/%:
	@if [ -f "$(REPLICATION_DIR)/$*/$(notdir $*).py" ]; then \
		echo "Running replication for $*"; \
		python $(REPLICATION_DIR)/$*/$(notdir $*).py; \
		touch $@; \
	else \
		echo "ERROR: Script not found at $(REPLICATION_DIR)/$*/$(notdir $*).py"; \
		exit 1; \
	fi

# Individual targets for each replication
.PHONY: $(REPLICATIONS)
$(REPLICATIONS): %: $(MARKER_DIR)/%

# Clean target
.PHONY: clean
clean:
	rm -rf $(MARKER_DIR)

# Special debug target for specific replication
.PHONY: debug_suarez
debug_suarez:
	@echo "Checking for suarezserratoWhoBenefitsState2016 files:"
	@ls -la $(REPLICATION_DIR)/suarezserratoWhoBenefitsState2016/ || echo "Directory not found"
	@echo "Looking for script file:"
	@find $(REPLICATION_DIR)/suarezserratoWhoBenefitsState2016/ -name "*.py" -type f

