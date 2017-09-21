close all;
clear;
clc;

%%
model = 'Fashion-MNIST';

results = [];

loaded = false;

for bits = 12:12:48
    mAPs = [];
    for iter = 10:10:100
        fpath = strcat(model, '-model-CPU-',num2str(iter),'-b',num2str(bits),'-data.mat')
        load(fpath);

        B_train = logical(B_train);
        B_test = logical(B_test);
        if loaded == false
            %%
            train_L = single(train_L);
            test_L = single(test_L);

            %% 
            S = compute_S (train_L,test_L);
            loaded = true;
        else
            clear train_L, test_L;
        end
        map = return_map (B_train, B_test, S)
        mAPs = [mAPs map];
    end
    results = [results; mAPs];
end

save(strcat('DSH-', model, '.mat'), 'results');

x = 10:10:100;
y = results;

figure;
plot(x, y,'-o','Linewidth',2);
title(['Deep Supervised Hashing Result on ', model]);
xlabel('Epochs');
ylabel('mAP');
grid on;
set(gca,'xtick',x);
legend('12bits','24bits','36bits','48bits');
drawnow;