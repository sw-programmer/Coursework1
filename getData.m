function [ data_train, data_query ] = getData( MODE )
% Generate training and testing data

% Data Options:
%   1. Toy_Gaussian
%   2. Toy_Spiral
%   3. Toy_Circle
%   4. Caltech 101

showImg = 0; % Show training & testing images and their image feature vector (histogram representation)

PHOW_Sizes = [4 8 10]; % Multi-resolution, these values determine the scale of each layer.
PHOW_Step = 8; % The lower the denser. Select from {2,4,8,16}

switch MODE
    case 'Toy_Gaussian' % Gaussian distributed 2D points
        %rand('state', 0);
        %randn('state', 0);
        N= 150;
        D= 2;
        
        cov1 = randi(4);
        cov2 = randi(4);
        cov3 = randi(4);
        
        X1 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov1 0;0 cov1]);
        X2 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov2 0;0 cov2]);
        X3 = mgd(N, D, [randi(4)-1 randi(4)-1], [cov3 0;0 cov3]);
        
        X= real([X1; X2; X3]);
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Spiral' % Spiral (from Karpathy's matlab toolbox)
        
        N= 50;
        t = linspace(0.5, 2*pi, N);
        x = t.*cos(t);
        y = t.*sin(t);
        
        t = linspace(0.5, 2*pi, N);
        x2 = t.*cos(t+2);
        y2 = t.*sin(t+2);
        
        t = linspace(0.5, 2*pi, N);
        x3 = t.*cos(t+4);
        y3 = t.*sin(t+4);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        X= bsxfun(@rdivide, bsxfun(@minus, X, mean(X)), var(X));
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Toy_Circle' % Circle
        
        N= 50;
        t = linspace(0, 2*pi, N);
        r = 0.4
        x = r*cos(t);
        y = r*sin(t);
        
        r = 0.8
        t = linspace(0, 2*pi, N);
        x2 = r*cos(t);
        y2 = r*sin(t);
        
        r = 1.2;
        t = linspace(0, 2*pi, N);
        x3 = r*cos(t);
        y3 = r*sin(t);
        
        X= [[x' y']; [x2' y2']; [x3' y3']];
        Y= [ones(N, 1); ones(N, 1)*2; ones(N, 1)*3];
        
        data_train = [X Y];
        
    case 'Caltech' % Caltech dataset
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));   % 15???????????? img
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); 
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [frames, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        disp(size(desc_tr))
        disp(size(desc_tr{1,1}))
        
        disp('Building visual codebook...')
        % Build visual vocabulary (codebook) for 'Bag-of-Words method'
        desc_sel = single(vl_colsubset(cat(2,desc_tr{:}), 10e4)); % Randomly select 100k SIFT descriptors for clustering
        disp(size(desc_sel)) 
        % K-means clustering
        numBins = 256; % for instance,
        
        
        % write your own codes here
        % [idx,centers] = kmeans(desc_sel,numBins);
        vocab = vl_kmeans(desc_sel, numBins)
        
        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        hists = {};
        for c = 1:length(classList)                       
            for i = 1:length(imgIdx_tr) % 15 imgs              
                [~, binsa] = min(vl_alldist2(vocab, single(desc_tr{c,i})), [], 1)  % codeword????????? ???????? ??????
                % frequency counting
                hists{c,i} = zeros(1,numBins)
                for j = 1:size(binsa,2)
                    hists{c,i}(1,binsa(1,j)) = hists{c,i}(1,binsa(1,j)) + 1
                end
            end
        end

        disp(size(desc_tr{1,1}))
        disp(sum(hists{1,1},2)) % hist ???????????? vocab size?????? ?????????????????? ???????????? ????????????
   
        data_train = hists
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
    
    case 'Caltech_RF' % Caltech dataset with RF codeword
        close all;
        imgSel = [15 15]; % randomly select 15 images each class without replacement. (For both training & testing)
        folderName = './Caltech_101/101_ObjectCategories';
        classList = dir(folderName);
        classList = {classList(3:end).name} % 10 classes
        
        disp('Loading training images...')
        % Load Images -> Description (Dense SIFT)
        cnt = 1;
        if showImg
            figure('Units','normalized','Position',[.05 .1 .4 .9]);
            suptitle('Training image samples');
        end
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx{c} = randperm(length(imgList));
            imgIdx_tr = imgIdx{c}(1:imgSel(1));   % 15???????????? img
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel)); 
            
            for i = 1:length(imgIdx_tr)
                I = imread(fullfile(subFolderName,imgList(imgIdx_tr(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I); % PHOW work on gray scale image
                end
                
                % For details of image description, see http://www.vlfeat.org/matlab/vl_phow.html
                [frames, desc_tr{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step); %  extracts PHOW features (multi-scaled Dense SIFT)
            end
        end
        disp(size(desc_tr))
        disp(size(desc_tr{1,1}))
        
        disp('Building RF codebook...')
        desc_fin = [];
        for i=1:10
            for j=1:15
                desc_tr{i,j} = cat(2,desc_tr{i,j},i);
                desc_fin = cat(1,desc_fin,desc_tr{i,j});
            end
        end

        [N,D] = size(desc_fin);
        frac = 1; % Bootstrap sampling fraction
        [labels,~] = unique(desc_fin(:,end));
        disp(labels)
        % Plot first 4 out of all data subsets
        for T = 1:4
            idx = randsample(N,ceil(N*frac),1); % A new training set for each trees is generated by random sampling from dataset WITH replacement.
            prior = histc(desc_fin(idx,end),labels)/length(idx);
        
        T = 1; % Tree number
        param.splitNum = 3; % Number of trials in split function

        ig_best = -inf;

        for n = 1:param.splitNum
            dim = randi(D-1);                           % Pick one random dimension as a split function
            d_min = single(min(data_train(idx,dim)));   % Find the data range of this dimension
            d_max = single(max(data_train(idx,dim)));
            t = d_min + rand*((d_max-d_min));           % Pick a random value within the range as threshold
            
            idx_ = data_train(idx,dim) < t;             % Split data with this dimension and threshold
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Calculate Information Gain
            
            L = data_train(idx_,:);
            R = data_train(~idx_,:);
            H = getE(data_train);        % Calculate entropy
            HL = getE(L);
            HR = getE(R);
            
            ig = H - sum(idx_)/length(idx_)*HL - sum(~idx_)/length(idx_)*HR
            
            if ig_best < ig
                ig_best = ig;   % maximu information gain saved
                t_best = t;     % the best threhold to save
                dim_best = dim; % the best split function (dimension) to save
                idx_best = idx_;
            end
            

            disp('Press any key to continue');
            pause; 
        end

        disp(size(data_train(idx,:)))
        disp(sum(idx_best))
        disp(dim_best)
        disp(t_best)
        disp(ig_best)

        param.depth = 5;        % Tree depth
        param.split = 'IG';
        
        % Initialise base node
        trees(T).node(1) = struct('idx',idx,'t',nan,'dim',-1,'prob',[]);
        % Split the nodes recursively
        for n = 1:2^(param.depth-1)-1
            [trees(T).node(n),trees(T).node(n*2),trees(T).node(n*2+1)] = splitNode(data_train,trees(T).node(n),param);
        end

        makeLeaf;
        visualise_leaf;

        vocab = []

        % Check size
        disp(size(trees(T).leaf(n).prob))

        for n = 1:2^param.depth-1
            vocab = cat(2,vocab,trees(T).leaf(n).prob);
        end

        disp('Encoding Images...')
        % Vector Quantisation
        
        % write your own codes here
        % ...
        hists = {};
        for c = 1:length(classList)                       
            for i = 1:length(imgIdx_tr) % 15 imgs              
                [~, binsa] = min(vl_alldist2(vocab, single(desc_tr{c,i})), [], 1)  % codeword????????? ???????? ??????
                % frequency counting
                hists{c,i} = zeros(1,numBins)
                for j = 1:size(binsa,2)
                    hists{c,i}(1,binsa(1,j)) = hists{c,i}(1,binsa(1,j)) + 1
                end
            end
        end

        disp(size(desc_tr{1,1}))
        disp(sum(hists{1,1},2)) % hist ???????????? vocab size?????? ?????????????????? ???????????? ????????????
   
        data_train = hists
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
end

end

switch MODE
    case 'Caltech'
        if showImg
        figure('Units','normalized','Position',[.05 .1 .4 .9]);
        suptitle('Test image samples');
        end
        disp('Processing testing images...');
        cnt = 1;
        % Load Images -> Description (Dense SIFT)
        for c = 1:length(classList)
            subFolderName = fullfile(folderName,classList{c});
            imgList = dir(fullfile(subFolderName,'*.jpg'));
            imgIdx_te = imgIdx{c}(imgSel(1)+1:sum(imgSel));
            
            for i = 1:length(imgIdx_te)
                I = imread(fullfile(subFolderName,imgList(imgIdx_te(i)).name));
                
                % Visualise
                if i < 6 & showImg
                    subaxis(length(classList),5,cnt,'SpacingVert',0,'MR',0);
                    imshow(I);
                    cnt = cnt+1;
                    drawnow;
                end
                
                if size(I,3) == 3
                    I = rgb2gray(I);
                end
                [~, desc_te{c,i}] = vl_phow(single(I),'Sizes',PHOW_Sizes,'Step',PHOW_Step);
            
            end
        end
        %suptitle('Testing image samples');
%                 if showImg
%             figure('Units','normalized','Position',[.5 .1 .4 .9]);
%         suptitle('Testing image representations: 256-D histograms');
%         end

        % Quantisation
        
        % write your own codes here
        % ...
        hists_test = {};
        for c = 1:length(classList)                       
            for i = 1:length(imgIdx_te) % 15 imgs              
                [~, binsa] = min(vl_alldist2(vocab, single(desc_te{c,i})), [], 1)  % codeword????????? ???????? ??????
                % frequency counting
                hists_test{c,i} = zeros(1,numBins)
                for j = 1:size(binsa,2)
                    hists_test{c,i}(1,binsa(1,j)) = hists_test{c,i}(1,binsa(1,j)) + 1
                end
            end
        end

        disp(size(desc_te{1,1}))
        disp(sum(hists_test{1,1},2)) % hist ???????????? vocab size?????? ?????????????????? ???????????? ????????????
   
        data_test = hists_test
        % Clear unused varibles to save memory
        clearvars desc_tr desc_sel
        
        
    otherwise % Dense point for 2D toy data
        xrange = [-1.5 1.5];
        yrange = [-1.5 1.5];
        inc = 0.02;
        [x, y] = meshgrid(xrange(1):inc:xrange(2), yrange(1):inc:yrange(2));
        data_query = [x(:) y(:) zeros(length(x)^2,1)];
end
end

