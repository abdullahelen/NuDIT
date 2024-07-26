%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The NuDIT v1.0
% Transforming Numerical Data to Images for Deep Networks.
%
% Assoc. Prof. Dr. Abdullah Elen
% Department of Software Engineering, Bandirma Onyedi Eylul University
% aelen@bandirma.edu.tr
% Date Created: 2024-07-25
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc; clear;
close all;

%% Phase #1: Initialize the NuDIT.
tabularFile = 'TabularDatasets\RiceMSCDataset.csv';
nudit = NuDIT(tabularFile);

%% Phase #2: Transforming numerical data to images.
nudit.numToImgTransform();

%% Phase #3: Apply k-fold cross-validation to dataset.
imageSize = 32;
kfold = 5;
nudit.prepareData(imageSize, kfold);

%% Phase #4: Start DAG-Net
epochs = 10;
result = nudit.runNetwork(epochs);
