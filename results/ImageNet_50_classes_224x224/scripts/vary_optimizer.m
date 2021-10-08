graph_title_text = "Varying Optimizer";

% exp _1
backprop_train_acc_1 = [0.924, 0.948, 0.951, 0.953, 0.954, 0.954, 0.957, 0.957, 0.958, 0.958, 0.958, 0.959, 0.958, 0.958, 0.96, 0.959, 0.96, 0.959, 0.96, 0.958]
backprop_val_acc_1   = [0.918, 0.919, 0.924, 0.924, 0.923, 0.921, 0.921, 0.923, 0.922, 0.922, 0.921, 0.917, 0.918, 0.918, 0.923, 0.921, 0.923, 0.919, 0.922, 0.921]
pc_train_acc_1       = [0.921, 0.952, 0.955, 0.957, 0.959, 0.96, 0.962, 0.963, 0.964, 0.965, 0.965, 0.967, 0.967, 0.968, 0.967, 0.968, 0.969, 0.968, 0.969, 0.97]
pc_val_acc_1         = [0.921, 0.925, 0.925, 0.923, 0.923, 0.924, 0.923, 0.924, 0.924, 0.923, 0.922, 0.922, 0.921, 0.922, 0.923, 0.919, 0.924, 0.923, 0.924, 0.922]

% exp _2
backprop_train_acc_2 = [0.907, 0.934, 0.94, 0.941, 0.945, 0.947, 0.948, 0.949, 0.95, 0.952, 0.954, 0.954, 0.954, 0.955, 0.955, 0.956, 0.957, 0.958, 0.959, 0.958]
backprop_val_acc_2   = [0.898, 0.907, 0.908, 0.909, 0.914, 0.913, 0.916, 0.915, 0.916, 0.916, 0.913, 0.911, 0.915, 0.917, 0.912, 0.914, 0.907, 0.915, 0.915, 0.914]
pc_train_acc_2       = [0.89, 0.922, 0.927, 0.931, 0.933, 0.934, 0.937, 0.938, 0.939, 0.94, 0.941, 0.942, 0.942, 0.943, 0.944, 0.944, 0.943, 0.945, 0.945, 0.945]
pc_val_acc_2         = [0.894, 0.896, 0.894, 0.892, 0.897, 0.896, 0.896, 0.896, 0.897, 0.896, 0.892, 0.896, 0.893, 0.893, 0.895, 0.895, 0.895, 0.897, 0.896, 0.898]

% exp _3
backprop_train_acc_3 = [0.874, 0.946, 0.952, 0.955, 0.958, 0.959, 0.959, 0.96, 0.961, 0.963, 0.964, 0.964, 0.964, 0.966, 0.965, 0.966, 0.966, 0.966, 0.966, 0.967]
backprop_val_acc_3   = [0.915, 0.919, 0.92, 0.925, 0.927, 0.924, 0.927, 0.928, 0.927, 0.926, 0.927, 0.926, 0.926, 0.927, 0.928, 0.926, 0.928, 0.927, 0.928, 0.927]
pc_train_acc_3       = [0.826, 0.935, 0.942, 0.944, 0.948, 0.949, 0.95, 0.952, 0.951, 0.954, 0.956, 0.955, 0.956, 0.957, 0.957, 0.958, 0.958, 0.959, 0.96, 0.959]
pc_val_acc_3         = [0.903, 0.909, 0.913, 0.918, 0.918, 0.918, 0.921, 0.922, 0.923, 0.924, 0.922, 0.922, 0.923, 0.924, 0.925, 0.923, 0.924, 0.924, 0.923, 0.923]

% exp _4
backprop_train_acc_4 = [0.742, 0.927, 0.937, 0.941, 0.945, 0.946, 0.95, 0.951, 0.951, 0.953, 0.954, 0.955, 0.954, 0.957, 0.957, 0.958, 0.957, 0.957, 0.958, 0.958]
backprop_val_acc_4   = [0.892, 0.907, 0.911, 0.916, 0.917, 0.92, 0.921, 0.922, 0.923, 0.923, 0.923, 0.924, 0.926, 0.926, 0.926, 0.927, 0.928, 0.927, 0.927, 0.927]
pc_train_acc_4       = [0.582, 0.889, 0.916, 0.927, 0.933, 0.935, 0.939, 0.939, 0.94, 0.942, 0.944, 0.944, 0.945, 0.943, 0.945, 0.947, 0.946, 0.946, 0.947, 0.946]
pc_val_acc_4         = [0.839, 0.881, 0.897, 0.905, 0.909, 0.91, 0.912, 0.914, 0.917, 0.914, 0.913, 0.918, 0.915, 0.919, 0.918, 0.916, 0.92, 0.92, 0.918, 0.919]


subplot(1,2,1);

% Backpropagation 
% Val accuracy
plot(backprop_val_acc_1*100, 'LineWidth',2)
hold on

plot(backprop_val_acc_2*100, 'LineWidth',2)
hold on

plot(backprop_val_acc_3*100, 'LineWidth',2);
hold on

plot(backprop_val_acc_4*100, 'LineWidth',2);
hold on


% Train accuracy
plot(backprop_train_acc_1*100, ':','LineWidth',2);
hold on

plot(backprop_train_acc_2*100, ':','LineWidth',2);
hold on

plot(backprop_train_acc_3*100, ':','LineWidth',2);
hold on

plot(backprop_train_acc_4*100, ':','LineWidth',2);
hold off


title('Backpropagation Accuracy', 'FontSize', 13)
xlabel('Epoch', 'FontSize', 13) 
ylabel('Accuracy (perc.)', 'FontSize', 13)

% Set plot 1 ranges
set(gca,'XTick',1:2:20);
axis([1 20 82 100])


subplot(1,2,2);

% PC 
% Valid accuracy
plot(pc_val_acc_1*100,'LineWidth',2);
hold on

plot(pc_val_acc_2*100,'LineWidth',2);
hold on

plot(pc_val_acc_3*100,'LineWidth',2);
hold on

plot(pc_val_acc_4*100,'LineWidth',2);
hold on


% Train accuracy
plot(pc_train_acc_1*100, ':','LineWidth',2);
hold on

plot(pc_train_acc_2*100, ':','LineWidth',2);
hold on

plot(pc_train_acc_3*100, ':','LineWidth',2);
hold on

plot(pc_train_acc_4*100, ':','LineWidth',2);
hold off


legend('Adam + Sigmoid (val)', 'Adam + Relu (val)','SGD + Relu (val)','SGD + Sigmoid (val)', 'Adam + Sigmoid (train)', 'Adam + Relu (train)','SGD + Relu (train)','SGD + Sigmoid (train)', 'FontSize', 12, 'Location','southeast')

title('Predictive Coding Accuracy', 'FontSize', 13)
xlabel('Epoch', 'FontSize', 13) 
ylabel('Loss', 'FontSize', 13) 

% Set plot 2 ranges
set(gca,'XTick',1:2:20);
axis([1 20 82 100])


% No scientific notation
% h =  plot(Valid_loss_feat, ':','LineWidth',2);
% ax = ancestor(h, 'axes');
% ax.XAxis.Exponent = 0;
xtickformat('%.0f'); 

% Padronize window size
x0=100;
y0=50;
width=1200;
height=600;
set(gcf,'position',[x0,y0,width,height])

sgtitle(graph_title_text, 'FontSize', 15) 


