#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run all analyses on integrated_data.csv.

This script sequentially executes data loading, preprocessing,
exploratory data analysis, and statistical modeling using the
existing modules in this repository.
"""

import data_exploration_fixed as de
import exploratory_data_analysis_fixed as eda
import statistical_modeling_fixed as sm


def main():
    # Step 1: Load and preprocess the raw integrated data
    data = de.load_data('integrated_data.csv')
    data = de.explore_data(data)

    # Step 2: Perform exploratory data analysis
    eda.exploratory_data_analysis(data)

    # Step 3: Run statistical modeling
    sm.statistical_modeling(data)


if __name__ == "__main__":
    main()
