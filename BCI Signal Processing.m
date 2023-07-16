                                        %% BSS - Project
                                        % Reza Jahani - 810198377
                    %% Choosing the subject
subj_number = 1;            
data = load('dataset/subj_1.mat').data;
                    %% Initial Section
data_c1 = data{1};
data_c2 = data{2};
data_c3 = data{3};
data_c4 = data{4};
[~,~,N1] = size(data_c1);
[~,~,N2] = size(data_c2);
[~,~,N3] = size(data_c3);
[~,~,N4] = size(data_c4);
data_c1 = zero_mean(data_c1);
data_c2 = zero_mean(data_c2);
data_c3 = zero_mean(data_c3);
data_c4 = zero_mean(data_c4);
data_channels = [39 43 15 44 16 45 17 46 18 47 19 48 20 49 21 50 22 51 23 52 24 53 25 54 26 55 27 56 28 57 29 58 30 59 31 60 32];
data_channels = sort(data_channels); 
data_channels = 1:63;   % subj 2 3 4 5 6 7 8 9 10 11 13 14
data_c1 = data_c1(data_channels,:,:);
data_c2 = data_c2(data_channels,:,:);
data_c3 = data_c3(data_channels,:,:);
data_c4 = data_c4(data_channels,:,:);
[M,T,~] = size(data_c1);

                    %% Hyperparameters and tuning
num_iteration = 20;
classifier_order = [4 3 2 1];
num_filters = [7 3 15]; % 7 10 15   % 8 8 8
filter_order = 5;
filter_bands = {[20 40;10 18;0.01 4] [20 40;10 15;0.1 8] [20 40;5 20;1 9] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [10 50;6 25;0.05 8] [5 60;6 30;0.5 5] [10 30;8 20;0.1 7] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4] [20 40;10 18;0.01 4]}; 
cl_1 = [classifier_order(1) 0 0;classifier_order(2) classifier_order(3) classifier_order(4)];
cl_2 = [classifier_order(2) 0;classifier_order(3) classifier_order(4)];
cl_3 = [classifier_order(3);classifier_order(4)];
cl = {cl_1 cl_2 cl_3};

                    %% Testing Phase
tic
correct_estimation_test = 0;
correct_estimation_train = 0;
confusion_matrix_test = zeros(4,4);
confusion_matrix_train = zeros(4,4);
for i=1:num_iteration
    test_idx_c1 = randi([1 N1]);
    test_idx_c2 = randi([1 N2]);
    test_idx_c3 = randi([1 N3]);
    test_idx_c4 = randi([1 N4]);
    c1_idx = 1:N1;
    c2_idx = 1:N2;
    c3_idx = 1:N3;
    c4_idx = 1:N4;
    X_c1_train = data_c1(:,:,c1_idx~=test_idx_c1);
    X_c2_train = data_c2(:,:,c2_idx~=test_idx_c2);
    X_c3_train = data_c3(:,:,c3_idx~=test_idx_c3);
    X_c4_train = data_c4(:,:,c4_idx~=test_idx_c4);
    X_c1_test = data_c1(:,:,test_idx_c1);
    X_c2_test = data_c2(:,:,test_idx_c2);
    X_c3_test = data_c3(:,:,test_idx_c3);
    X_c4_test = data_c4(:,:,test_idx_c4);
    [Wcsp,W_LDA,C] = Model_Train(X_c1_train,X_c2_train,X_c3_train,X_c4_train,M,cl,num_filters,filter_order,filter_bands,subj_number);
        % Test
    y1 = Model_Test(Wcsp,W_LDA,C,X_c1_test,classifier_order,filter_order,filter_bands,subj_number);
    y2 = Model_Test(Wcsp,W_LDA,C,X_c2_test,classifier_order,filter_order,filter_bands,subj_number);
    y3 = Model_Test(Wcsp,W_LDA,C,X_c3_test,classifier_order,filter_order,filter_bands,subj_number);
    y4 = Model_Test(Wcsp,W_LDA,C,X_c4_test,classifier_order,filter_order,filter_bands,subj_number);
    predicted_label = [y1 y2 y3 y4];
    true_label = [1 2 3 4];
    for m=1:4
       confusion_matrix_test(true_label(m),predicted_label(m)) = confusion_matrix_test(true_label(m),predicted_label(m)) + 1;  
       if true_label(m) == predicted_label(m)
          correct_estimation_test = correct_estimation_test + 1; 
       end
    end
        % Validation
    N_max = max([N1-1 N2-1 N3-1 N4-1]);
    for n=1:N_max
        if n <= N1 -1
            X_c1_test = X_c1_train(:,:,n);
            y1 = Model_Test(Wcsp,W_LDA,C,X_c1_test,classifier_order,filter_order,filter_bands,subj_number);
            if y1==1
               correct_estimation_train = correct_estimation_train + 1; 
            end
            confusion_matrix_train(1,y1) = confusion_matrix_train(1,y1) + 1;
        end
        if n <= N2 -1
            X_c2_test = X_c2_train(:,:,n);
            y2 = Model_Test(Wcsp,W_LDA,C,X_c2_test,classifier_order,filter_order,filter_bands,subj_number);
            if y2==2
               correct_estimation_train = correct_estimation_train + 1; 
            end
            confusion_matrix_train(2,y2) = confusion_matrix_train(2,y2) + 1;
        end
        if n <= N3 -1
            X_c3_test = X_c3_train(:,:,n);
            y3 = Model_Test(Wcsp,W_LDA,C,X_c3_test,classifier_order,filter_order,filter_bands,subj_number);
            if y3==3
               correct_estimation_train = correct_estimation_train + 1; 
            end
            confusion_matrix_train(3,y3) = confusion_matrix_train(3,y3) + 1;            
        end
        if n <= N4 -1
            X_c4_test = X_c4_train(:,:,n);
            y4 = Model_Test(Wcsp,W_LDA,C,X_c4_test,classifier_order,filter_order,filter_bands,subj_number);
            if y4==4
               correct_estimation_train = correct_estimation_train + 1; 
            end
            confusion_matrix_train(4,y4) = confusion_matrix_train(4,y4) + 1;            
        end
    end
end
accuracy_test = correct_estimation_test/(num_iteration*4);
accuracy_train = correct_estimation_train/(num_iteration*(N1+N2+N3+N4-4));

toc

disp('Confusion Matrix Test = ');
disp(confusion_matrix_test./sum(confusion_matrix_test,2));
disp('Confusion Matrix Train = ');
disp(confusion_matrix_train./sum(confusion_matrix_train,2));

            %% Local Necessary Functions 
    
function [Wcsp,W_LDA,c] = Model_Train(data_c1,data_c2,data_c3,data_c4,M,cl,num_filters,filter_order,filter_bands,subj_number)
    W = filter_bands{subj_number};
    w1 = W(1,:);
    w2 = W(2,:);
    w3 = W(3,:);
    X_c1_train = data_c1;
    X_c2_train = data_c2;
    X_c3_train = data_c3;
    X_c4_train = data_c4;
    X_ac = {X_c1_train X_c2_train X_c3_train X_c4_train};

                % Ordered Classifiers
    cl_1 = cl{1};
    cl_2 = cl{2};
    cl_3 = cl{3};
        % Classifier 1
    X1 = X_ac{cl_1(1,1)};
    X2 = cat(3,X_ac{cl_1(2,1)},X_ac{cl_1(2,2)},X_ac{cl_1(2,3)});
    X1 = filter_bp(X1,w1,filter_order);
    X2 = filter_bp(X2,w1,filter_order);
    Wcsp_1 = W_csp(X1,X2,M,num_filters(1));
    [Xc1,~] = CSP(Wcsp_1,X1);
    [Xc234,~] = CSP(Wcsp_1,X2);
    [c1,W_LDA_1] = Get_WLDA(Xc1,Xc234);

        % Classifier 2
    X1 = X_ac{cl_2(1,1)};
    X2 = cat(3,X_ac{cl_2(2,1)},X_ac{cl_2(2,2)});
    X1 = filter_bp(X1,w2,filter_order);
    X2 = filter_bp(X2,w2,filter_order);
    Wcsp_2 = W_csp(X1,X2,M,num_filters(2));
    [Xc2,~] = CSP(Wcsp_2,X1);
    [Xc34,~] = CSP(Wcsp_2,X2);
    [c2,W_LDA_2] = Get_WLDA(Xc2,Xc34);

        % Classifier 3
    X1 = X_ac{cl_3(1,1)};
    X2 = X_ac{cl_3(2,1)};
    X1 = filter_bp(X1,w3,filter_order);
    X2 = filter_bp(X2,w3,filter_order);
    Wcsp_3 = W_csp(X1,X2,M,num_filters(3));
    [Xc3,~] = CSP(Wcsp_3,X1);
    [Xc4,~] = CSP(Wcsp_3,X2);
    [c3,W_LDA_3] = Get_WLDA(Xc3,Xc4);

    Wcsp = {Wcsp_1 Wcsp_2 Wcsp_3};
    W_LDA = {W_LDA_1 W_LDA_2 W_LDA_3};
    c = [c1 c2 c3];
end

function y = Model_Test(W_csp,W_LDA,C,X_test,classifier_order,filter_order,filter_bands,subj_number)
    W = filter_bands{subj_number};
    for i=1:3
       w = W(i,:);
       X = filter_bp(X_test,w,filter_order);
       Wcsp = W_csp{i};
       Wlda = W_LDA{i};
       c = C(i);
       Y = CSP_Model(Wcsp,Wlda,c,X);
       if Y==1
          y = classifier_order(i);
          break;
       end
    end
    if Y==0
       y = classifier_order(4); 
    end
end
    
function y = zero_mean(x)
    y = x - mean(x,2);
end

function Rx = AutoCorr(X)
    size_X = size(X);
    N = size_X(3);
    M = size_X(1);
    Rx = zeros(M,M);
    for n=1:N
       x = X(:,:,n);
       Rx = Rx + x*transpose(x);
    end
    Rx = 1/N * Rx;
end

function Wcsp = W_csp(X1,X2,M,m)
    Rx1 = AutoCorr(X1);
    Rx2 = AutoCorr(X2);
    [Wcsp,Dcsp] = eig(Rx1,Rx2);
    Dcsp_vals = diag(Dcsp);
    Dcsp_vals = sort(Dcsp_vals,'descend');
    Wcsp_temp = Wcsp;
    for i=1:M
        [~,col] = find(Dcsp==Dcsp_vals(i)); 
        Wcsp(:,i) = Wcsp_temp(:,col);
    end
    Wcsp = normc(Wcsp);
    main_filter = [1:m M-m+1:M];
    Wcsp = Wcsp(:,main_filter);
end

function [Var_Mat,Y] = CSP(Wcsp,X)
    [~,T,N] = size(X);
    [~,L] = size(Wcsp);
    Y = zeros(L,T,N);
    Var_Mat = zeros(L,N);
    for n=1:N
       x = X(:,:,n);
       Y(:,:,n) = transpose(Wcsp) * x;
       Var_Mat(:,n) = var(transpose(Y(:,:,n)));
    end
end

function C = Cov(Xc,m)
    [M,N] = size(Xc);
    C = zeros(M,M);
    for i=1:N
        C = C + (Xc(:,i)-m)*transpose((Xc(:,i)-m));
    end
    C = 1/N * C;
end

function [c,W_LDA] = Get_WLDA(Xc1,Xc2)
    m1 = mean(Xc1,2);
    m2 = mean(Xc2,2);
    C1 = Cov(Xc1,m1); 
    C2 = Cov(Xc2,m2);
    Mat1 = (m1-m2)*transpose(m1-m2);
    Mat2 = C1 + C2;
    [Q,D] = eig(Mat1,Mat2);
    [~,idx] = max(max(abs(D)));
    W_LDA = Q(:,idx);
    miu_1 = transpose(W_LDA)*m1;
    miu_2 = transpose(W_LDA)*m2;
    c = 1/2 * (miu_1 + miu_2);
    if miu_1 < miu_2
        W_LDA = -W_LDA;
        c = - c;
    end
end

function Y = CSP_Model(Wcsp,Wlda,c,X_test)
    X = X_test;
    [x,~] = CSP(Wcsp,X);
    y = transpose(Wlda) * x;
    Y(y>=c) = 1;
    Y(y<c) = 0;
end

function Y = filter_bp(X,w,order)
    fs = 2400;
    [zz,pp,kk] = butter(order,w/fs/2,'bandpass');
    sos = zp2sos(zz,pp,kk);
    [M,T,N] = size(X);
    Y = zeros(M,T,N);
    for n=1:N
       x = X(:,:,n);
       x = transpose(x);
       y = sosfilt(sos,x);
       Y(:,:,n) = transpose(y);
    end
end