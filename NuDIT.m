%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The NuDIT v1.0
% Transforming Numerical Data to Images for Deep Networks.
%
% Assoc. Prof. Dr. Abdullah Elen
% Department of Software Engineering, Bandirma Onyedi Eylul University
% aelen@bandirma.edu.tr
% Date Created: 2024-07-25
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef NuDIT < handle

    properties
        DataStore;
        DatasetName;
        CsvFile;
        Labels;
        Partitions;
        ImageSize;
    end

    methods (Access = public)
        function obj = NuDIT(csvFile)
            % Extract file name for dataset name.
            obj.CsvFile = csvFile;
            [~, obj.DatasetName, ~] = fileparts(csvFile);

            clc;
            % Show message.
            disp(' ');
            disp([' <strong>Welcome to the NuDIT</strong>' '']);
            disp(' Transforming Numerical Data to Images for Deep Networks.');
            disp(' Programmed by Abdullah Elen, 2024.');
            disp('______________________________________________________________________________');
            disp(' ');
        end

        function prepareData(obj, imageSize, kfold)
            disp([' <strong># Preparing ', num2str(kfold), ...
                '-fold cross-validation data..</strong>']);

            % Set size of image.
            obj.ImageSize = imageSize;

            % Datastore for image dataset.
            imgsetPath = fullfile('Exports', obj.DatasetName);
            dataStore = imageDatastore(imgsetPath, ... % Dataset folder
                'LabelSource', 'foldernames', 'IncludeSubfolders', true);

            % k-fold cross validation.
            obj.Partitions = cvpartition(dataStore.Labels, 'Kfold', kfold);

            % Convert gray-scale and resize images.
            dataStore.ReadFcn = @(x) imresize(...
                im2gray(imread(x)), [obj.ImageSize, obj.ImageSize]);

            obj.DataStore = dataStore;

            % Görüntü verisetine ait etiketler ve sayılarının gösterilmesi.
            obj.Labels = countEachLabel(dataStore);
            disp([9, 'The image dataset is ready for training!']);
            disp(' ');
        end

        function numToImgTransform(obj)
            disp(' <strong># Transformation process has started..</strong>');
            disp(' ');
            % Create data table from file.
            dataOriginal = readtable(obj.CsvFile);

            % Get features and class labels from the dataset.
            inputs = table2array(dataOriginal(:, 1:end-1));
            outputs = dataOriginal(:, end);
            outputs = categorical(outputs{:, 1});

            clsLabels = unique(outputs);
            if isnumeric(clsLabels)
                clsLabels = num2str(clsLabels);
            end

            % Build path of the image dataset.
            dirRoot = fullfile('Exports', obj.DatasetName);

            % Create directory for the image dataset.
            if ~exist(dirRoot, 'dir'), mkdir(dirRoot); end

            % Create directory for each class.
            for i = 1 : numel(clsLabels)
                dirClass = fullfile(dirRoot, string(clsLabels(i)));
                if ~exist(dirClass, 'dir'), mkdir(dirClass); end
            end

            % Normalize dataset.
            dataNorm = normalize(inputs, 'range');

            % Calculate the size of image.
            dataWidth = size(dataNorm, 2);
            sqrSize = ceil(sqrt(dataWidth));
            diffSize = sqrSize^2 - dataWidth;

            if (diffSize > 0)
                rowSize = size(dataNorm, 1);
                dataNorm = [dataNorm, zeros(rowSize, diffSize)];
            end

            % Init. progress bar.
            hWaitBar = waitbar(0, 'Processing...');

            for i = 1 : numel(clsLabels)
                indices = (outputs == clsLabels(i));
                dataList = dataNorm(indices, :);
                totalSteps = size(dataList, 1);

                for j = 1 : totalSteps
                    % Update progress bar.
                    ratio = j / totalSteps;
                    waitbar(ratio, hWaitBar, ...
                        sprintf('Transforming (%d of %d): %3.1f%%', ...
                        i, numel(clsLabels), ratio * 100));

                    data = dataList(j, :);
                    % Reshape image array into sqrSize-by-sqrSize matrix.
                    dataMatrix = reshape(data, [sqrSize, sqrSize]);
                    % Convert matrix to grayscale image.
                    imgData = mat2gray(dataMatrix);
                    % Build full image file name.
                    imgFile = fullfile(dirRoot, string(clsLabels(i)), [num2str(j), '.png']);
                    % Save image file to target folder.
                    imwrite(imgData, imgFile, BitDepth=8);
                end
            end

            close(hWaitBar);

            % Display final message!
            disp([9, '<strong>', obj.DatasetName, '</strong>', ' image dataset is ready!']);
            for i = 1 : numel(clsLabels)
                total = sum(outputs == clsLabels(i));
                disp([9, num2str(i), ') ', char(clsLabels(i)), ': ', num2str(total)]);
            end
            disp([9, 'Check "Exports\', obj.DatasetName, '" to view the image files.']);
        end

        function result = getDataSet(obj)
            % Get image files from dataStore
            imgFiles = obj.DataStore.Files;

            % Get number of partition.
            partCount = obj.Partitions.NumTestSets;

            result.TrainSet = cell(partCount, 1);
            result.TestSet = cell(partCount, 1);

            for i = 1 : partCount
                % TrainSet
                imds = imageDatastore(imgFiles(training(obj.Partitions, i)), ...
                    'LabelSource', 'foldernames', 'IncludeSubfolders', true);
                imds.ReadFcn = obj.DataStore.ReadFcn;
                result.TrainSet{i} = imds;

                % TestSet
                imds = imageDatastore(imgFiles(test(obj.Partitions, i)), ...
                    'LabelSource', 'foldernames', 'IncludeSubfolders', true);
                imds.ReadFcn = obj.DataStore.ReadFcn;
                result.TestSet{i} = imds;
            end

            return;
        end

        function result = runNetwork(obj, epochs)
            disp(' <strong># The training of the DAG-Net has started..</strong>');
            disp(' ');
            % Training options for deep learning (Bu özellikleri değiştirebilirsin).
            trainOpts = trainingOptions('sgdm', ...
                'MiniBatchSize', 16, ... % Genelde bu değer 32, 64 ya da 128 oluyor.
                'MaxEpochs', epochs, ... % Genelde bu değer 30-50 arasında oluyor.
                'ExecutionEnvironment', 'cpu', ... % 'cpu' ya da 'gpu'
                'Plots', 'training-progress', ... % 'none' ya da 'training-progress'
                'Verbose', true);

            % "outputSize" derin öğrenme modellerinin çıkış sayısını, otomatikmen sınıf
            % sayısı olarak belirler.
            outputSize = size(obj.Labels, 1);
            inputSize = [obj.ImageSize, obj.ImageSize, 1];

            % Initialize DAG-Net
            model = obj.dagNet(inputSize, outputSize);

            % Partition data for cross-validation.
            parts = obj.Partitions;

            % Accuracy array.
            accuracy = zeros(parts.NumTestSets, 1);

            dataSet = obj.getDataSet();

            for j = 1 : parts.NumTestSets
                % Training process.
                trainSet = dataSet.TrainSet{j};
                network = trainNetwork(trainSet, model, trainOpts);                
                
                % Test process.
                testSet = dataSet.TestSet{j};
                [predictions, ~] = classify(network, testSet);

                % Display classification accuracy.
                truePreds = sum(predictions == testSet.Labels);
                accuracy(j) = truePreds / numel(testSet.Labels);

                disp(['* TEST RESULT => fold-#', num2str(j), ...
                    ': classification accuracy: %', num2str(accuracy(j)*100)]);
                disp(' ');
            end

            result = accuracy;
        end

        function result = dagNet(~, inputSize, outputSize)
            result = layerGraph();

            layers = [
                imageInputLayer(inputSize, 'Name', 'input')
                convolution2dLayer(5, 16, 'Padding', 'same', 'Name', 'conv_1')
                batchNormalizationLayer('Name', 'BN_1')
                reluLayer('Name', 'relu_1')
                convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', 2, 'Name', 'conv_2')
                batchNormalizationLayer('Name', 'BN_2')
                reluLayer('Name', 'relu_2')
                convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv_3')
                batchNormalizationLayer('Name', 'BN_3')
                reluLayer('Name', 'relu_3')
                additionLayer(2, 'Name', 'add')
                averagePooling2dLayer(2, 'Stride', 2, 'Name', 'avpool')
                fullyConnectedLayer(outputSize, 'Name', 'fcl')
                softmaxLayer('Name', 'softmax')
                classificationLayer('Name', 'classOutput')
                ];

            result = addLayers(result, layers);

            layers = convolution2dLayer(1, 32, 'Stride', 2, 'Name', 'skipConv');
            result = addLayers(result, layers);

            result = connectLayers(result, 'relu_1', 'skipConv');
            result = connectLayers(result, 'skipConv', 'add/in2');
        end
    end
end
