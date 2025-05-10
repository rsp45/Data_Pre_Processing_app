document.addEventListener('DOMContentLoaded', function() {
    let currentStep = 1;
    let dataAnalysis = null;
    let modelType = null;
    let problemType = null;

    // Navigation buttons
    const prevStepBtn = document.getElementById('prevStep');
    const nextStepBtn = document.getElementById('nextStep');

    prevStepBtn.addEventListener('click', () => {
        if (currentStep > 1) {
            goToStep(currentStep - 1);
        }
    });

    nextStepBtn.addEventListener('click', () => {
        if (currentStep < 4) {
            goToStep(currentStep + 1);
        }
    });

    // File Upload Handling
    const fileInput = document.getElementById('fileInput');
    fileInput.addEventListener('change', handleFileUpload);

    // Drag and Drop
    const dropZone = document.querySelector('.border-dashed');
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-purple-600');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-purple-600');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-purple-600');
        fileInput.files = e.dataTransfer.files;
        handleFileUpload();
    });

    // Test Size Slider
    const testSizeSlider = document.getElementById('testSize');
    const testSizeValue = document.getElementById('testSizeValue');
    testSizeSlider.addEventListener('input', () => {
        testSizeValue.textContent = testSizeSlider.value;
    });

    // Advanced EDA Button
    document.getElementById('performAdvancedEDA').addEventListener('click', performAdvancedEDA);

    // Hyperparameter Tuning Toggle
    const hyperparameterToggle = document.getElementById('enableHyperparameterTuning');
    const hyperparameterOptions = document.getElementById('hyperparameterOptions');
    hyperparameterToggle.addEventListener('change', () => {
        hyperparameterOptions.classList.toggle('hidden', !hyperparameterToggle.checked);
    });

    // Model Training
    document.getElementById('trainButton').addEventListener('click', trainModel);
    document.getElementById('saveModelButton').addEventListener('click', saveModel);
    document.getElementById('downloadPlotsButton').addEventListener('click', downloadPlots);
    document.getElementById('compareModelsBtn').addEventListener('click', compareModels);

    // Preprocessing
    document.getElementById('preprocessButton').addEventListener('click', performPreprocessing);
    document.getElementById('testSize').addEventListener('input', function() {
        document.getElementById('testSizeValue').textContent = this.value;
    });

    // Initialize example datasets
    loadExampleDatasets();

    async function handleFileUpload() {
        try {
            const file = fileInput.files[0];
            if (!file) {
                showError('Please select a file to upload');
                return;
            }

            if (!file.name.endsWith('.csv')) {
                showError('Only CSV files are supported');
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            showLoading('Uploading file...');
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.status === 'error') {
                throw new Error(result.message);
            }

            showSuccess('File uploaded successfully!');
            
            // Update UI with file info
            const fileInfo = document.createElement('div');
            fileInfo.className = 'mt-4 p-4 bg-gray-50 rounded-lg';
            fileInfo.innerHTML = `
                <h4 class="font-medium mb-2">Dataset Information</h4>
                <p>Rows: ${result.shape[0]}</p>
                <p>Columns: ${result.shape[1]}</p>
                <p class="mt-2 font-medium">Available Columns:</p>
                <ul class="list-disc pl-5">
                    ${result.columns.map(col => `<li>${col}</li>`).join('')}
                </ul>
            `;
            document.querySelector('.border-dashed').appendChild(fileInfo);

            // Update target column dropdown
            const targetSelect = document.getElementById('targetColumn');
            targetSelect.innerHTML = result.columns
                .map(col => `<option value="${col}">${col}</option>`)
                .join('');

            // Enable and show the next step
            goToStep(2);
            hideLoading();
        } catch (error) {
            hideLoading();
            showError('Error uploading file: ' + error.message);
        }
    }

    async function handleDataUpload(event) {
        event.preventDefault();
        
        try {
            const formData = new FormData(document.getElementById('uploadForm'));
            showLoading('Uploading data...');
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            if (result.status === 'error') {
                throw new Error(result.message);
            }
            
            // Store columns for later use
            window.availableColumns = result.columns;
            
            // Populate target and feature selectors
            populateColumnSelectors(result.columns);
            
            // Show target selection step
            goToStep(2);
            showSuccess('Data uploaded successfully!');
            hideLoading();
        } catch (error) {
            hideLoading();
            showError('Upload Error: ' + error.message);
        }
    }

    function populateColumnSelectors(columns) {
        const targetSelect = document.getElementById('targetColumn');
        targetSelect.innerHTML = '';
        
        // Add default option
        const defaultOption = document.createElement('option');
        defaultOption.value = '';
        defaultOption.textContent = 'Select target column';
        targetSelect.appendChild(defaultOption);
        
        // Add column options
        columns.forEach(column => {
            const option = document.createElement('option');
            option.value = column;
            option.textContent = column;
            targetSelect.appendChild(option);
        });
        
        // Add event listener for target selection
        targetSelect.addEventListener('change', async function() {
            if (this.value) {
                await handleTargetSelection(this.value);
            }
        });
    }

    async function handleTargetSelection(targetColumn) {
        try {
            showLoading('Analyzing data...');

            // Set target column
            const response = await fetch('/set_target', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ target_column: targetColumn })
            });

            const result = await response.json();
            if (result.status === 'error') {
                throw new Error(result.message);
            }

            // Perform EDA
            const edaResponse = await fetch('/advanced_eda');
            const edaResults = await edaResponse.json();
            
            if (edaResults.status === 'error') {
                throw new Error(edaResults.message);
            }

            // Display EDA results
            displayEDAResults(edaResults);
            
            // Automatically proceed with preprocessing
            const preprocessResponse = await fetch('/preprocess_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            const preprocessResult = await preprocessResponse.json();
            if (preprocessResult.status === 'error') {
                throw new Error(preprocessResult.message);
            }

            displayPreprocessingResults(preprocessResult);
            goToStep(3);

        } catch (error) {
            showError('Error performing advanced EDA: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    function displayEDAResults(results) {
        const edaSection = document.getElementById('edaResults');
        edaSection.innerHTML = '';

        // Create container for visualizations
        const visualContainer = document.createElement('div');
        visualContainer.className = 'grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8';
        edaSection.appendChild(visualContainer);

        // Correlation Matrix
        if (results.correlation_plot) {
            const correlationDiv = document.createElement('div');
            correlationDiv.className = 'bg-white p-4 rounded-lg shadow col-span-2';
            correlationDiv.style.height = '800px';
            visualContainer.appendChild(correlationDiv);

            Plotly.newPlot(
                correlationDiv,
                results.correlation_plot.data,
                results.correlation_plot.layout
            );
        }

        // Distribution Plots for numeric features
        if (results.distribution_plots) {
            Object.entries(results.distribution_plots).forEach(([col, plot]) => {
                const plotDiv = document.createElement('div');
                plotDiv.className = 'bg-white p-4 rounded-lg shadow';
                plotDiv.style.height = '400px';
                visualContainer.appendChild(plotDiv);

                Plotly.newPlot(
                    plotDiv,
                    plot.data,
                    plot.layout
                );
            });
        }

        // Box Plots
        if (results.box_plots) {
            const boxPlotDiv = document.createElement('div');
            boxPlotDiv.className = 'bg-white p-4 rounded-lg shadow col-span-2';
            boxPlotDiv.style.height = `${Math.max(400, Object.keys(results.distribution_plots).length * 50)}px`;
            visualContainer.appendChild(boxPlotDiv);

            Plotly.newPlot(
                boxPlotDiv,
                results.box_plots.data,
                results.box_plots.layout
            );
        }

        // Categorical Plots
        if (results.categorical_plots) {
            Object.entries(results.categorical_plots).forEach(([col, plot]) => {
                const plotDiv = document.createElement('div');
                plotDiv.className = 'bg-white p-4 rounded-lg shadow';
                plotDiv.style.height = '400px';
                visualContainer.appendChild(plotDiv);

                Plotly.newPlot(
                    plotDiv,
                    plot.data,
                    plot.layout
                );
            });
        }

        // Numeric Statistics Table
        if (results.numeric_stats) {
            const statsContainer = document.createElement('div');
            statsContainer.className = 'bg-white p-4 rounded-lg shadow col-span-2 overflow-x-auto';
            
            const table = document.createElement('table');
            table.className = 'min-w-full divide-y divide-gray-200';
            
            // Create header
            const thead = document.createElement('thead');
            thead.className = 'bg-gray-50';
            const headerRow = document.createElement('tr');
            headerRow.innerHTML = `
                <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Statistic</th>
                ${Object.keys(results.numeric_stats).map(col => 
                    `<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">${col}</th>`
                ).join('')}
            `;
            thead.appendChild(headerRow);
            table.appendChild(thead);

            // Create body
            const tbody = document.createElement('tbody');
            tbody.className = 'bg-white divide-y divide-gray-200';
            const stats = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'];
            stats.forEach(stat => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${stat}</td>
                    ${Object.values(results.numeric_stats).map(col => 
                        `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${col[stat]}</td>`
                    ).join('')}
                `;
                tbody.appendChild(row);
            });
            table.appendChild(tbody);
            
            statsContainer.appendChild(table);
            edaSection.appendChild(statsContainer);
        }
    }

    function displayPostEDA(edaResults) {
        const postEdaResults = document.getElementById('postEdaResults');
        postEdaResults.innerHTML = '';

        const edaSection = document.createElement('div');
        edaSection.className = 'mb-8 p-6 bg-white rounded-lg shadow';
        
        let content = `
            <h3 class="text-xl font-semibold mb-4">Post-Processing Analysis</h3>
            
            <!-- Target Distribution -->
            <div class="mb-6">
                <h4 class="text-lg font-medium mb-3">Target Distribution</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${edaResults.problem_type === 'classification' ? `
                        <!-- Classification Target Distribution -->
                        <div class="p-4 bg-blue-50 rounded-lg">
                            <h5 class="font-medium mb-2">Training Set</h5>
                            <div class="space-y-2">
                                ${Object.entries(edaResults.target_distribution.train)
                                    .map(([className, percentage]) => `
                                        <div class="flex justify-between items-center">
                                            <span>Class ${className}</span>
                                            <span class="font-medium">${(percentage * 100).toFixed(2)}%</span>
                                        </div>
                                    `).join('')}
                            </div>
                        </div>
                        <div class="p-4 bg-green-50 rounded-lg">
                            <h5 class="font-medium mb-2">Test Set</h5>
                            <div class="space-y-2">
                                ${Object.entries(edaResults.target_distribution.test)
                                    .map(([className, percentage]) => `
                                        <div class="flex justify-between items-center">
                                            <span>Class ${className}</span>
                                            <span class="font-medium">${(percentage * 100).toFixed(2)}%</span>
                                        </div>
                                    `).join('')}
                            </div>
                        </div>
                    ` : `
                        <!-- Regression Target Distribution -->
                        <div class="p-4 bg-blue-50 rounded-lg">
                            <h5 class="font-medium mb-2">Training Set</h5>
                            <div class="space-y-2">
                                <div class="flex justify-between">
                                    <span>Mean</span>
                                    <span class="font-medium">${edaResults.target_distribution.train.mean.toFixed(3)}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Std</span>
                                    <span class="font-medium">${edaResults.target_distribution.train.std.toFixed(3)}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Range</span>
                                    <span class="font-medium">${edaResults.target_distribution.train.min.toFixed(3)} - ${edaResults.target_distribution.train.max.toFixed(3)}</span>
                                </div>
                            </div>
                        </div>
                        <div class="p-4 bg-green-50 rounded-lg">
                            <h5 class="font-medium mb-2">Test Set</h5>
                            <div class="space-y-2">
                                <div class="flex justify-between">
                                    <span>Mean</span>
                                    <span class="font-medium">${edaResults.target_distribution.test.mean.toFixed(3)}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Std</span>
                                    <span class="font-medium">${edaResults.target_distribution.test.std.toFixed(3)}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Range</span>
                                    <span class="font-medium">${edaResults.target_distribution.test.min.toFixed(3)} - ${edaResults.target_distribution.test.max.toFixed(3)}</span>
                                </div>
                            </div>
                        </div>
                    `}
                </div>
            </div>

            ${edaResults.feature_importance ? `
                <!-- Feature Importance -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium mb-3">Feature Importance</h4>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead>
                                <tr>
                                    <th class="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase">Feature</th>
                                    <th class="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase">Importance</th>
                                </tr>
                            </thead>
                            <tbody class="bg-white divide-y divide-gray-200">
                                ${Object.entries(edaResults.feature_importance)
                                    .sort((a, b) => b[1] - a[1])
                                    .map(([feature, importance]) => `
                                        <tr>
                                            <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${feature}</td>
                                            <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                <div class="flex items-center">
                                                    <div class="w-48 bg-gray-200 rounded-full h-2.5">
                                                        <div class="bg-blue-600 h-2.5 rounded-full" style="width: ${(importance * 100).toFixed(1)}%"></div>
                                                    </div>
                                                    <span class="ml-2">${(importance * 100).toFixed(1)}%</span>
                                                </div>
                                            </td>
                                        </tr>
                                    `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            ` : ''}

            ${Object.keys(edaResults.correlations).length > 0 ? `
                <!-- Feature Correlations -->
                <div class="mb-6">
                    <h4 class="text-lg font-medium mb-3">Feature Correlations</h4>
                    <div class="overflow-x-auto">
                        <table class="min-w-full divide-y divide-gray-200">
                            <thead>
                                <tr>
                                    <th class="px-6 py-3 bg-gray-50"></th>
                                    ${edaResults.correlations.features.map(feature => `
                                        <th class="px-6 py-3 bg-gray-50 text-left text-xs font-medium text-gray-500 uppercase">${feature}</th>
                                    `).join('')}
                                </tr>
                            </thead>
                            <tbody class="bg-white divide-y divide-gray-200">
                                ${edaResults.correlations.features.map(feature1 => `
                                    <tr>
                                        <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${feature1}</td>
                                        ${edaResults.correlations.features.map(feature2 => {
                                            const correlation = edaResults.correlations.matrix[feature1][feature2];
                                            const bgColor = correlation > 0 ? 
                                                `rgba(59, 130, 246, ${Math.abs(correlation)})` : 
                                                `rgba(239, 68, 68, ${Math.abs(correlation)})`;
                                            return `
                                                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500" 
                                                    style="background-color: ${bgColor}">
                                                    ${correlation.toFixed(2)}
                                                </td>
                                            `;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            ` : ''}

            <!-- Feature Distributions -->
            <div>
                <h4 class="text-lg font-medium mb-3">Feature Distributions</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    ${Object.entries(edaResults.distributions).map(([feature, distribution]) => `
                        <div class="p-4 bg-gray-50 rounded-lg">
                            <h5 class="font-medium mb-2">${feature}</h5>
                            <div class="space-y-4">
                                <div>
                                    <h6 class="text-sm font-medium text-blue-700">Training Set</h6>
                                    <div class="space-y-1">
                                        <div class="flex justify-between text-sm">
                                            <span>Mean: ${distribution.train.mean.toFixed(3)}</span>
                                            <span>Std: ${distribution.train.std.toFixed(3)}</span>
                                        </div>
                                        <div class="flex justify-between text-sm">
                                            <span>Min: ${distribution.train.min.toFixed(3)}</span>
                                            <span>Max: ${distribution.train.max.toFixed(3)}</span>
                                        </div>
                                    </div>
                                </div>
                                <div>
                                    <h6 class="text-sm font-medium text-green-700">Test Set</h6>
                                    <div class="space-y-1">
                                        <div class="flex justify-between text-sm">
                                            <span>Mean: ${distribution.test.mean.toFixed(3)}</span>
                                            <span>Std: ${distribution.test.std.toFixed(3)}</span>
                                        </div>
                                        <div class="flex justify-between text-sm">
                                            <span>Min: ${distribution.test.min.toFixed(3)}</span>
                                            <span>Max: ${distribution.test.max.toFixed(3)}</span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;

        edaSection.innerHTML = content;
        postEdaResults.appendChild(edaSection);
    }

    function updateDataInsightsUI(insights) {
        const insightsContainer = document.getElementById('dataInsights');
        const recommendationsContainer = document.getElementById('dataRecommendations');

        // Update basic stats
        insightsContainer.innerHTML = `
            <div class="p-4 bg-gray-50 rounded-lg">
                <h4 class="font-medium mb-2">Basic Statistics</h4>
                <p>Total Rows: ${insights.basic_stats.total_rows}</p>
                <p>Total Columns: ${insights.basic_stats.total_columns}</p>
                <p>Missing Data: ${insights.basic_stats.missing_percentage.toFixed(2)}%</p>
            </div>
            <div class="p-4 bg-gray-50 rounded-lg">
                <h4 class="font-medium mb-2">Column Types</h4>
                <p>Numerical Columns: ${insights.column_types.numerical}</p>
                <p>Categorical Columns: ${insights.column_types.categorical}</p>
            </div>
        `;

        // Update recommendations
        recommendationsContainer.innerHTML = insights.recommendations
            .map(rec => `
                <div class="p-3 rounded-lg ${rec.type === 'warning' ? 'bg-yellow-50 text-yellow-800' : 'bg-blue-50 text-blue-800'}">
                    <i class="mdi ${rec.type === 'warning' ? 'mdi-alert' : 'mdi-information'} mr-2"></i>
                    ${rec.message}
                </div>
            `)
            .join('');
    }

    function updateAnalysisUI(result) {
        // Update target column dropdown
        const targetSelect = document.getElementById('targetColumn');
        targetSelect.innerHTML = result.columns
            .map(col => `<option value="${col}">${col}</option>`)
            .join('');

        // Show analysis section and hide upload section
        document.getElementById('edaResults').classList.remove('hidden');
    }

    async function performAdvancedEDA() {
        try {
            showLoading('Performing advanced EDA...');
            
            const response = await fetch('/advanced_eda');
            const results = await response.json();
            
            if (results.status === 'error') {
                throw new Error(results.message);
            }

            displayEDAResults(results);
            showSuccess('Advanced EDA completed successfully!');
            
        } catch (error) {
            console.error('Error:', error);
            showError('Error performing advanced EDA: ' + error.message);
        } finally {
            hideLoading();
        }
    }

    async function performPreprocessing() {
        try {
            const targetSelect = document.getElementById('targetColumn');
            if (!targetSelect || !targetSelect.value) {
                throw new Error('Please select a target column first');
            }

            showLoading('Setting target column and preprocessing data...');
            
            // First ensure target is set
            const targetResponse = await fetch('/set_target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target_column: targetSelect.value })
            });
            
            if (!targetResponse.ok) throw new Error('Failed to set target column');
            const targetResult = await targetResponse.json();
            
            if (targetResult.status === 'success') {
                // Now proceed with preprocessing
                const preprocessResponse = await fetch('/preprocess_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });
                
                if (!preprocessResponse.ok) throw new Error('Failed to preprocess data');
                const preprocessResults = await preprocessResponse.json();
                
                // Display preprocessing results
                displayPreprocessingResults(preprocessResults);
                
                hideLoading();
                showSuccess('Data preprocessing completed successfully!');
                goToStep(3); // Move to model training step
            } else {
                throw new Error(targetResult.message || 'Failed to set target column');
            }
        } catch (error) {
            hideLoading();
            showError(error.message);
        }
    }

    async function handlePreprocessing() {
        await performPreprocessing();
    }

    function displayPreprocessingResults(results) {
        const preprocessingSection = document.getElementById('preprocessingResults');
        if (!preprocessingSection) return;

        // Clear previous results
        preprocessingSection.innerHTML = '';

        // Create preprocessing summary
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'bg-white p-6 rounded-lg shadow-md mb-6';
        
        // Add preprocessing steps
        if (results.steps && results.steps.length > 0) {
            const stepsDiv = document.createElement('div');
            stepsDiv.className = 'mb-4';
            stepsDiv.innerHTML = `
                <h3 class="text-lg font-semibold mb-2">Preprocessing Steps:</h3>
                <ul class="list-disc pl-5 space-y-1">
                    ${results.steps.map(step => `<li class="text-gray-700">${step}</li>`).join('')}
                </ul>
            `;
            summaryDiv.appendChild(stepsDiv);
        }

        // Add data shapes
        if (results.shapes) {
            const shapesDiv = document.createElement('div');
            shapesDiv.className = 'mb-4';
            shapesDiv.innerHTML = `
                <h3 class="text-lg font-semibold mb-2">Data Shapes:</h3>
                <div class="grid grid-cols-3 gap-4">
                    <div class="bg-gray-50 p-3 rounded">
                        <p class="font-medium">Original</p>
                        <p class="text-gray-700">${results.shapes.original.join(' × ')}</p>
                    </div>
                    <div class="bg-gray-50 p-3 rounded">
                        <p class="font-medium">Training Set</p>
                        <p class="text-gray-700">${results.shapes.train.join(' × ')}</p>
                    </div>
                    <div class="bg-gray-50 p-3 rounded">
                        <p class="font-medium">Test Set</p>
                        <p class="text-gray-700">${results.shapes.test.join(' × ')}</p>
                    </div>
                </div>
            `;
            summaryDiv.appendChild(shapesDiv);
        }

        // Add feature information
        if (results.feature_names) {
            const featuresDiv = document.createElement('div');
            featuresDiv.className = 'mb-4';
            featuresDiv.innerHTML = `
                <h3 class="text-lg font-semibold mb-2">Features (${results.feature_names.length}):</h3>
                <div class="bg-gray-50 p-3 rounded">
                    <p class="text-gray-700">${results.feature_names.join(', ')}</p>
                </div>
            `;
            summaryDiv.appendChild(featuresDiv);
        }

        // Add target distribution for classification problems
        if (results.target_distribution) {
            const distributionDiv = document.createElement('div');
            distributionDiv.className = 'mb-4';
            distributionDiv.innerHTML = `
                <h3 class="text-lg font-semibold mb-2">Target Distribution:</h3>
                <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    ${Object.entries(results.target_distribution).map(([label, count]) => `
                        <div class="bg-gray-50 p-3 rounded">
                            <p class="font-medium">Class ${label}</p>
                            <p class="text-gray-700">${count} samples</p>
                        </div>
                    `).join('')}
                </div>
            `;
            summaryDiv.appendChild(distributionDiv);
        }

        preprocessingSection.appendChild(summaryDiv);

        // Add "Continue to Model Training" button
        const continueButton = document.createElement('button');
        continueButton.className = 'bg-purple-600 text-white px-4 py-2 rounded-lg hover:bg-purple-700 mt-4';
        continueButton.textContent = 'Continue to Model Training';
        continueButton.onclick = () => goToStep(3);
        preprocessingSection.appendChild(continueButton);
    }

    // Model descriptions
    const MODEL_DESCRIPTIONS = {
        'rf': 'Random Forest: A versatile ensemble method that builds multiple decision trees for robust predictions.',
        'et': 'Extra Trees: Similar to Random Forest but with more randomization for better generalization.',
        'dt': 'Decision Tree: A simple yet interpretable model that makes decisions based on feature thresholds.',
        'gb': 'Gradient Boosting: Builds trees sequentially, each correcting errors from previous trees.',
        'xgb': 'XGBoost: An optimized gradient boosting implementation known for its speed and performance.',
        'lgb': 'LightGBM: A fast gradient boosting framework that uses tree-based learning algorithms.',
        'cat': 'CatBoost: Gradient boosting library with built-in categorical feature support.',
        'ada': 'AdaBoost: Focuses on difficult training examples by adjusting their weights.',
        'bag': 'Bagging: Builds multiple models on random subsets of data for stable predictions.',
        'lr': 'Linear/Logistic Regression: Simple but interpretable models for regression/classification.',
        'ridge': 'Ridge Regression: Linear regression with L2 regularization to prevent overfitting.',
        'lasso': 'Lasso Regression: Linear regression with L1 regularization for feature selection.',
        'svm': 'Support Vector Machine: Finds the optimal hyperplane to separate classes.',
        'knn': 'K-Nearest Neighbors: Makes predictions based on similar training examples.'
    };

    // Update model selection UI based on problem type
    function updateModelSelectionUI(problemType) {
        const modelCheckboxes = document.querySelectorAll('.model-checkbox');
        const regressionOnlyModels = document.querySelectorAll('.regression-only');
        
        modelCheckboxes.forEach(checkbox => {
            // Enable all checkboxes
            checkbox.disabled = false;
            
            // Add tooltips
            const label = checkbox.nextElementSibling;
            label.setAttribute('title', MODEL_DESCRIPTIONS[checkbox.value]);
            
            // Add Bootstrap tooltip
            new bootstrap.Tooltip(label);
        });

        // Handle regression-only models
        if (problemType === 'classification') {
            regressionOnlyModels.forEach(model => {
                model.style.display = 'none';
            });
        } else {
            regressionOnlyModels.forEach(model => {
                model.style.display = 'block';
            });
        }

        // Enable compare button if any models are selected
        updateCompareButton();
    }

    // Update compare button state
    function updateCompareButton() {
        const selectedModels = document.querySelectorAll('.model-checkbox:checked');
        const compareButton = document.getElementById('compareModelsBtn');
        compareButton.disabled = selectedModels.length === 0;
    }

    // Add event listeners for model checkboxes
    document.querySelectorAll('.model-checkbox').forEach(checkbox => {
        checkbox.addEventListener('change', updateCompareButton);
    });

    // Compare selected models
    async function compareModels() {
        // Get all trained model checkboxes
        const modelCheckboxes = document.querySelectorAll('.model-checkbox:checked');
        const selectedModels = Array.from(modelCheckboxes).map(cb => cb.value);

        if (selectedModels.length < 2) {
            showError('Please select at least two models to compare');
            return;
        }

        console.log('Selected models for comparison:', selectedModels);

        try {
            showLoading('Comparing models...');
            const response = await fetch('/compare_models', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_types: selectedModels
                })
            });

            const results = await response.json();
            hideLoading();

            if (results.status === 'success') {
                displayComparisonResults(results);
            } else {
                showError(results.message || 'Error comparing models');
            }
        } catch (error) {
            hideLoading();
            console.error('Comparison error:', error);
            showError('Error comparing models: ' + error.message);
        }
    }

    async function trainModel() {
        try {
            // Get selected model type
            const modelTypeSelect = document.getElementById('modelType');
            const selectedModel = modelTypeSelect.value;

            if (!selectedModel) {
                showError('Please select a model type');
                return;
            }

            console.log('Selected model:', selectedModel); // Debug log

            showLoading('Training model...');
            const response = await fetch('/train_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_type: selectedModel
                })
            });

            const results = await response.json();
            console.log('Training results:', results); // Debug log
            hideLoading();
            
            if (results.status === 'success') {
                displayModelResults(results);
            } else {
                showError(results.message || 'Error training model');
            }
        } catch (error) {
            hideLoading();
            console.error('Training error:', error); // Debug log
            showError('Error training model: ' + error.message);
        }
    }

    function displayModelResults(results) {
        console.log('Raw results received:', results);
        
        const modelResults = document.getElementById('modelResults');
        modelResults.innerHTML = '';

        if (!results || results.status === 'error') {
            console.error('Error in results:', results);
            showError(results.message || 'Error training model');
            return;
        }

        // Add model information
        const modelInfo = document.createElement('div');
        modelInfo.className = 'bg-gray-50 p-4 rounded-lg mb-8';

        // Create shape info
        let shapeInfo = '';
        if (results.data_shapes && results.data_shapes.train_shape && results.data_shapes.test_shape) {
            const trainShape = results.data_shapes.train_shape;
            const testShape = results.data_shapes.test_shape;
            
            shapeInfo = `
                <p class="mb-2"><span class="font-medium">Training Data:</span> ${trainShape[0]} samples × ${trainShape[1]} features</p>
                <p class="mb-2"><span class="font-medium">Testing Data:</span> ${testShape[0]} samples × ${testShape[1]} features</p>
            `;
        } else {
            shapeInfo = '<p class="text-yellow-600">Shape information not available</p>';
        }

        modelInfo.innerHTML = `
            <h3 class="text-lg font-semibold mb-2">Model Information</h3>
            <p class="mb-2"><span class="font-medium">Model Type:</span> ${results.model_type}</p>
            ${shapeInfo}
            <p class="text-sm text-gray-600">${MODEL_DESCRIPTIONS[results.model_type] || ''}</p>
        `;
        modelResults.appendChild(modelInfo);

        // Display metrics if available
        if (results.metrics) {
            const metricsSection = document.createElement('div');
            metricsSection.className = 'mb-8';
            metricsSection.innerHTML = '<h3 class="text-lg font-semibold mb-4">Model Performance Metrics</h3>';

            const metricsGrid = document.createElement('div');
            metricsGrid.className = 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4';

            Object.entries(results.metrics).forEach(([metric, value]) => {
                const card = document.createElement('div');
                card.className = 'bg-green-50 p-4 rounded-lg';
                card.innerHTML = `
                    <h4 class="text-sm font-medium text-green-800">${formatMetricName(metric)}</h4>
                    <p class="text-2xl font-bold text-green-900">${formatMetricValue(value)}</p>
                `;
                metricsGrid.appendChild(card);
            });

            metricsSection.appendChild(metricsGrid);
            modelResults.appendChild(metricsSection);
        }

        // Display feature importance if available
        if (results.feature_importance && results.feature_importance.data) {
            const plotContainer = document.createElement('div');
            plotContainer.className = 'mb-8';
            plotContainer.innerHTML = `
                <h3 class="text-lg font-semibold mb-4">Feature Importance</h3>
                <div id="featureImportancePlot"></div>
            `;
            modelResults.appendChild(plotContainer);

            const data = [{
                type: 'bar',
                x: Object.values(results.feature_importance.data),
                y: Object.keys(results.feature_importance.data),
                orientation: 'h',
                marker: {
                    color: 'rgb(79, 70, 229)'
                }
            }];

            const layout = {
                title: 'Feature Importance',
                xaxis: { title: 'Importance Score' },
                yaxis: { title: 'Feature' },
                margin: { l: 200 }
            };

            Plotly.newPlot('featureImportancePlot', data, layout);
        }

        // Show model comparison section
        const comparisonSection = document.createElement('div');
        comparisonSection.className = 'mt-8 p-4 bg-white rounded-lg shadow';
        comparisonSection.innerHTML = `
            <h3 class="text-lg font-semibold mb-4">Model Comparison</h3>
            <p class="mb-4">Select models to compare with the current model:</p>
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
                ${Object.entries(MODEL_DESCRIPTIONS).map(([key, description]) => `
                    <div class="form-check">
                        <input type="checkbox" class="form-check-input model-checkbox" id="${key}" value="${key}">
                        <label class="form-check-label" for="${key}" title="${description}">
                            ${formatModelName(key)}
                        </label>
                    </div>
                `).join('')}
            </div>
            <div class="text-right">
                <button id="compareModelsBtn" class="btn btn-primary" disabled>
                    Compare Selected Models
                </button>
            </div>
        `;
        modelResults.appendChild(comparisonSection);

        // Add event listeners for the new checkboxes
        document.querySelectorAll('.model-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', updateCompareButton);
        });

        // Add custom tooltip behavior
        document.querySelectorAll('.form-check-label').forEach(label => {
            const description = label.getAttribute('title');
            if (description) {
                label.removeAttribute('title'); // Remove title to prevent default tooltip
                
                // Show tooltip on hover
                label.addEventListener('mouseenter', (e) => {
                    const tooltip = document.createElement('div');
                    tooltip.className = 'tooltip position-absolute bg-dark text-white p-2 rounded';
                    tooltip.style.cssText = 'z-index: 1000; max-width: 300px; font-size: 0.875rem;';
                    tooltip.textContent = description;
                    document.body.appendChild(tooltip);
                    
                    // Position tooltip
                    const rect = label.getBoundingClientRect();
                    tooltip.style.left = rect.left + 'px';
                    tooltip.style.top = (rect.top - tooltip.offsetHeight - 5) + 'px';
                    
                    // Store tooltip reference
                    label.tooltip = tooltip;
                });
                
                // Hide tooltip
                label.addEventListener('mouseleave', () => {
                    if (label.tooltip) {
                        label.tooltip.remove();
                        label.tooltip = null;
                    }
                });
            }
        });

        // Enable save model button
        document.getElementById('saveModelButton').disabled = false;

        showSuccess(results.message || 'Model training completed successfully!');
        
        // Move to evaluation step
        goToStep(4);
    }

    // Format model name for display
    function formatModelName(key) {
        const names = {
            'rf': 'Random Forest',
            'et': 'Extra Trees',
            'dt': 'Decision Tree',
            'gb': 'Gradient Boosting',
            'xgb': 'XGBoost',
            'lgb': 'LightGBM',
            'cat': 'CatBoost',
            'ada': 'AdaBoost',
            'bag': 'Bagging',
            'lr': 'Linear/Logistic',
            'ridge': 'Ridge',
            'lasso': 'Lasso',
            'svm': 'SVM',
            'knn': 'KNN'
        };
        return names[key] || key;
    }

    // Model comparison functionality
    async function compareModels() {
        // Get all trained model checkboxes
        const modelCheckboxes = document.querySelectorAll('.model-checkbox:checked');
        const selectedModels = Array.from(modelCheckboxes).map(cb => cb.value);

        if (selectedModels.length < 2) {
            showError('Please select at least two models to compare');
            return;
        }

        console.log('Selected models for comparison:', selectedModels);

        try {
            showLoading('Comparing models...');
            const response = await fetch('/compare_models', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_types: selectedModels
                })
            });

            const results = await response.json();
            hideLoading();

            if (results.status === 'success') {
                displayComparisonResults(results);
            } else {
                showError(results.message || 'Error comparing models');
            }
        } catch (error) {
            hideLoading();
            console.error('Comparison error:', error);
            showError('Error comparing models: ' + error.message);
        }
    }

    function updateModelSelectionGrid(currentModel) {
        const grid = document.getElementById('modelSelectionGrid');
        if (!grid) return;

        // Enable the checkbox for the current model
        const checkbox = grid.querySelector(`input[value="${currentModel}"]`);
        if (checkbox) {
            checkbox.disabled = false;  // Start disabled until trained
            checkbox.checked = true;
        }

        // Enable compare button if at least two models are selected
        const selectedModels = grid.querySelectorAll('.model-checkbox:checked');
        const compareButton = document.getElementById('compareModelsBtn');
        if (compareButton) {
            compareButton.disabled = selectedModels.length < 2;
        }
    }

    function displayComparisonResults(results) {
        console.log('Comparison results:', results);
        
        const comparisonResults = document.getElementById('comparisonResults');
        comparisonResults.innerHTML = '';

        if (!results || results.status === 'error') {
            showError(results.message || 'Error comparing models');
            return;
        }

        // Create comparison section
        const comparisonSection = document.createElement('div');
        comparisonSection.className = 'bg-white p-6 rounded-lg shadow-lg mb-8';
        
        // Add header
        comparisonSection.innerHTML = `
            <h3 class="text-xl font-semibold mb-4">Model Comparison Results</h3>
            <div id="metricsComparison" class="grid grid-cols-1 gap-6"></div>
        `;
        comparisonResults.appendChild(comparisonSection);

        // Display metrics comparison
        const metricsComparison = document.getElementById('metricsComparison');
        
        // Create a plot for each metric
        if (results.plots) {
            Object.entries(results.plots).forEach(([metric, plotData]) => {
                const plotContainer = document.createElement('div');
                plotContainer.className = 'bg-gray-50 p-4 rounded-lg';
                plotContainer.innerHTML = `
                    <h4 class="text-lg font-medium mb-2">${formatMetricName(metric)}</h4>
                    <div id="plot_${metric}" class="w-full h-64"></div>
                `;
                metricsComparison.appendChild(plotContainer);

                // Create the plot
                const data = [{
                    type: 'bar',
                    x: plotData.x,
                    y: plotData.y,
                    marker: {
                        color: 'rgb(79, 70, 229)'
                    }
                }];

                const layout = {
                    title: plotData.title,
                    xaxis: {
                        title: 'Models',
                        tickangle: -45
                    },
                    yaxis: {
                        title: formatMetricName(metric),
                        range: metric.includes('accuracy') || metric.includes('r2') ? [0, 1] : null
                    },
                    margin: {
                        l: 50,
                        r: 50,
                        t: 50,
                        b: 100
                    }
                };

                Plotly.newPlot(
                    `plot_${metric}`,
                    data,
                    layout
                );
            });
        }

        // Create metrics table
        const metricsTable = document.createElement('div');
        metricsTable.className = 'mt-8';
        metricsTable.innerHTML = `
            <h4 class="text-lg font-medium mb-4">Detailed Metrics</h4>
            <div class="overflow-x-auto">
                <table class="min-w-full bg-white">
                    <thead class="bg-gray-50">
                        <tr>
                            <th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                            ${Object.keys(Object.values(results.comparison)[0].metrics).map(metric => 
                                `<th class="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">${formatMetricName(metric)}</th>`
                            ).join('')}
                        </tr>
                    </thead>
                    <tbody class="divide-y divide-gray-200">
                        ${Object.entries(results.comparison).map(([model, data]) => `
                            <tr>
                                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">${model}</td>
                                ${Object.values(data.metrics).map(value => 
                                    `<td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${formatMetricValue(value)}</td>`
                                ).join('')}
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `;
        comparisonResults.appendChild(metricsTable);

        showSuccess('Model comparison completed successfully!');
    }

    function getPrimaryMetric(comparison) {
        // Determine if it's classification or regression based on available metrics
        return Object.values(comparison.models)[0].hasOwnProperty('accuracy') ? 'accuracy' : 'r2';
    }

    function formatMetricName(metric) {
        const metricMap = {
            'accuracy': 'Accuracy',
            'precision': 'Precision',
            'recall': 'Recall',
            'f1': 'F1 Score',
            'r2': 'R² Score',
            'mse': 'Mean Squared Error',
            'mae': 'Mean Absolute Error',
            'rmse': 'Root Mean Squared Error'
        };
        return metricMap[metric] || metric;
    }

    function formatMetricValue(value) {
        if (typeof value === 'number') {
            // For error metrics (MSE, MAE, RMSE), show more decimal places
            if (value > 1) {
                return value.toFixed(4);
            }
            // For scores and percentages (accuracy, r2, etc.), show as percentage
            return (value * 100).toFixed(2) + '%';
        }
        return value;
    }

    async function saveModel() {
        try {
            showLoading('Saving model...');
            const response = await fetch('/save_model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_name: `model_${Date.now()}.joblib`
                })
            });

            const result = await response.json();
            if (result.status === 'error') throw new Error(result.message);

            showSuccess('Model saved successfully!');
            hideLoading();
        } catch (error) {
            hideLoading();
            showError('Error saving model: ' + error.message);
        }
    }

    async function downloadPlots() {
        try {
            const modelInsights = document.getElementById('modelInsights');
            const plots = modelInsights.getElementsByTagName('div');
            
            for (let plot of plots) {
                if (plot.firstElementChild && plot.firstElementChild.tagName === 'DIV') {
                    const plotElement = plot.firstElementChild;
                    const plotTitle = plot.querySelector('h3').textContent;
                    
                    const plotData = await Plotly.toImage(plotElement, {
                        format: 'png',
                        width: 1200,
                        height: 800
                    });
                    
                    const link = document.createElement('a');
                    link.href = plotData;
                    link.download = `${plotTitle.toLowerCase().replace(/\s+/g, '_')}.png`;
                    link.click();
                }
            }
            
            showSuccess('Plots downloaded successfully!');
        } catch (error) {
            showError('Error downloading plots: ' + error.message);
        }
    }

    async function tuneHyperparameters() {
        const nTrials = document.getElementById('nTrials').value;
        const cvFolds = document.getElementById('cvFolds').value;
        const cvStrategy = document.getElementById('cvStrategy').value;
        const modelType = document.getElementById('modelType').value;
        
        const data = {
            n_trials: parseInt(nTrials),
            cv_folds: parseInt(cvFolds),
            cv_strategy: cvStrategy,
            model_type: modelType
        };

        try {
            const response = await fetch('/tune_hyperparameters', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error('Hyperparameter tuning failed');
            }

            const result = await response.json();
            updateModelParameters(result.best_params);
            showNotification('Hyperparameter tuning completed successfully', 'success');
        } catch (error) {
            console.error('Error during hyperparameter tuning:', error);
            showNotification('Error during hyperparameter tuning: ' + error.message, 'error');
        }
    }

    function updateModelParameters(params) {
        Object.keys(params).forEach(param => {
            const element = document.getElementById(param);
            if (element) {
                element.value = params[param];
            }
        });
    }

    function goToStep(step) {
        // Update progress indicators
        document.querySelectorAll('.step').forEach((el, index) => {
            if (index + 1 <= step) {
                el.classList.add('active');
            } else {
                el.classList.remove('active');
            }
        });

        // Show/hide sections
        document.querySelectorAll('.section').forEach((el, index) => {
            if (index + 1 === step) {
                el.classList.remove('hidden');
                el.classList.add('active');
            } else {
                el.classList.add('hidden');
                el.classList.remove('active');
            }
        });

        // Update navigation buttons
        prevStepBtn.classList.toggle('hidden', step === 1);
        nextStepBtn.classList.toggle('hidden', step === 4);

        currentStep = step;
    }

    function showLoading(message) {
        Swal.fire({
            title: message,
            allowOutsideClick: false,
            showConfirmButton: false,
            willOpen: () => {
                Swal.showLoading();
            }
        });
    }

    function hideLoading() {
        Swal.close();
    }

    function showError(message) {
        Swal.fire({
            icon: 'error',
            title: 'Error',
            text: message
        });
    }

    function showSuccess(message) {
        Swal.fire({
            icon: 'success',
            title: 'Success',
            text: message
        });
    }

    async function loadExampleDatasets() {
        try {
            const response = await fetch('/available_datasets');
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.message);
            }

            const datasets = result.datasets;
            const datasetContainer = document.createElement('div');
            datasetContainer.className = 'mt-8 p-4 bg-white rounded-lg shadow';
            datasetContainer.innerHTML = `
                <h3 class="text-lg font-medium mb-4">Or try an example dataset:</h3>
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    ${Object.entries(datasets).map(([key, dataset]) => `
                        <div class="p-4 border rounded-lg hover:shadow-md cursor-pointer transition-all"
                             onclick="loadExampleDataset('${key}')">
                            <h4 class="font-medium text-lg">${dataset.name}</h4>
                            <p class="text-gray-600 text-sm mb-2">${dataset.description}</p>
                            <div class="flex justify-between text-sm text-gray-500">
                                <span>Type: ${dataset.type}</span>
                                <span>${dataset.features} features</span>
                                <span>${dataset.samples} samples</span>
                            </div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            document.querySelector('.upload-section').appendChild(datasetContainer);
        } catch (error) {
            showError('Error loading example datasets: ' + error.message);
        }
    }

    async function loadExampleDataset(datasetName) {
        try {
            showLoading('Loading dataset...');
            const response = await fetch(`/load_example_dataset/${datasetName}`);
            const result = await response.json();
            
            if (result.status === 'error') {
                throw new Error(result.message);
            }

            showSuccess(result.message);
            
            // Update UI with dataset info
            const fileInfo = document.createElement('div');
            fileInfo.className = 'mt-4 p-4 bg-gray-50 rounded-lg';
            fileInfo.innerHTML = `
                <h4 class="font-medium mb-2">Dataset Information</h4>
                <p>Rows: ${result.shape[0]}</p>
                <p>Columns: ${result.shape[1]}</p>
                <p class="mt-2 font-medium">Available Columns:</p>
                <ul class="list-disc pl-5">
                    ${result.columns.map(col => `<li>${col}</li>`).join('')}
                </ul>
            `;
            document.querySelector('.border-dashed').appendChild(fileInfo);

            // Update target column dropdown
            const targetSelect = document.getElementById('targetColumn');
            targetSelect.innerHTML = result.columns
                .map(col => `<option value="${col}">${col}</option>`)
                .join('');

            // Enable and show the next step
            goToStep(2);
            hideLoading();
        } catch (error) {
            hideLoading();
            showError('Error loading dataset: ' + error.message);
        }
    }

    document.addEventListener('DOMContentLoaded', () => {
        loadExampleDatasets();
        document.getElementById('compareModelsBtn').addEventListener('click', compareModels);
    });
});
