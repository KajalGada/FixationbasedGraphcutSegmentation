close all
clear all
clc

addpath(genpath('../maxflow-v3.01_matlab/'))

%Read image
imName = 'Images/DSC01527';
fix_x = 209;   % fixation point
fix_y = 145;   % fixation point

im = imread([imName, '.png']);
im = im2double(im);
edgeGrad = load([imName, '_grad.mat']);  %% Use precomputed Berkeley edges
edgeGrad = edgeGrad.edgeGrad;
edgeOri = load([imName, '_ori.mat']);
edgeOri = edgeOri.edgeOri;

%   im,             input image (must be double)
%   edgeGrad,       image probabilistic edge map
%   fix_x,          fixation point x coordinate
%   fix_y,          fixation point y coordinate

% Energy fucntion parameters

% Binary weights
nju = 5;         % Binary weight exponent constant
k = 20;          % Binary weight for two pixels with zero edge probability
lambda = 1000;   % Importance of binary weights

% Unary weights
D  = 1e100;      % Fixation point unary weight
D_ = 10;

%unaryColourWeight = 1;      % Importance of colourmodel unary weights

foreground = 2;  % Object
background = 1;  % Background

% Image parameters
[yRes, xRes] = size(edgeGrad);
nPix = xRes*yRes;

verbose = 1;

if verbose
    fprintf('Computing binary weight matrix\n');
end
E = edges4connected(yRes,xRes); % Indices of adjacent pixels (Potts model)

% Step 1: Average edge probability at adjacent edges
E_avg = (edgeGrad(E(:,1)) + edgeGrad(E(:,2))) / 2;   

% For Binary Weights
bw = zeros(size(E,1),1);

% Step 2: Edge where at least one of the pixels belongs to the edge map
bw(E_avg ~= 0) = exp( -nju*( E_avg(E_avg ~= 0) ));
% Step 3: Edges where none of the pixels belong to the edge map, assign k
bw(E_avg == 0) = k;  

% Step 4: Calculate the distance of each edge from the fixation point
[py, px] = ind2sub([yRes, xRes], E(:,1));
[qy, qx] = ind2sub([yRes, xRes], E(:,2));
x_mid = (px + qx) / 2 - fix_x(1);
y_mid = (py + qy) / 2 - fix_y(1);
r = sqrt(x_mid.^2 + y_mid.^2);

% Step 5: Weights are the inverse of the distance from the fixation point
wt = 1./r;
% Step 6: Normalize the weights to have maximum of 1
wt = wt/max(wt);
bw = bw.*wt;

A = sparse(E(:,1),E(:,2),bw,nPix,nPix,4*nPix);

% Step 7: Construct unary weights for image boundary and fixation point 
T_ = zeros(numel(edgeGrad),2);
T_(1:yRes,background) = D;             % Left column
T_(end-yRes+1:end,background) = D;     % Right column
T_((0:xRes-1)*yRes+1,background) =D;   % Top row
T_((1:xRes)*yRes,background) =D;       % Bottom row

% Fixation Point
T_(sub2ind([yRes, xRes], fix_y(1), fix_x(1)), foreground) = D;         
T_(sub2ind([yRes, xRes], fix_y(2:end), fix_x(2:end)), foreground) = D_;  
T = sparse(T_);

% Step 8: Perform min-cut
[flow, labels] = maxflowmex(A,T);
labels = reshape(labels, [yRes, xRes]);

% Show results
imgBoundary = bwmorph(labels==1, 'remove');
imgBoundary = bwmorph(imgBoundary, 'dilate');

figure;
imshow(imgBoundary);
title('Results');

fprintf('end');









