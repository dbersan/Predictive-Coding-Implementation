graph_title_text = "Varying Network Depth";


backprop_train_acc = [0.776, 0.807, 0.809, 0.807, 0.806, 0.804, 0.803, 0.801, 0.8, 0.799, 0.799, 0.799, 0.798, 0.798, 0.798, 0.797, 0.798, 0.797, 0.797, 0.798]
backprop_val_acc   = [0.766, 0.767, 0.761, 0.762, 0.758, 0.755, 0.755, 0.754, 0.752, 0.749, 0.755, 0.748, 0.749, 0.755, 0.751, 0.748, 0.747, 0.747, 0.747, 0.748]
pc_train_acc       = [0.736, 0.821, 0.83, 0.834, 0.837, 0.839, 0.841, 0.843, 0.844, 0.845, 0.847, 0.847, 0.848, 0.849, 0.849, 0.851, 0.851, 0.852, 0.852, 0.853]
pc_val_acc         = [0.772, 0.777, 0.779, 0.78, 0.781, 0.782, 0.783, 0.783, 0.784, 0.783, 0.783, 0.781, 0.782, 0.782, 0.782, 0.782, 0.782, 0.784, 0.782, 0.782]


% Backpropagation 
% Val accuracy
plot(backprop_val_acc*100, 'LineWidth',2)
hold on

% Train accuracy
plot(backprop_train_acc*100, ':','LineWidth',2);

hold on

% PC 
% Valid accuracy
plot(pc_val_acc*100,'LineWidth',2);
hold on

% Train accuracy
plot(pc_train_acc*100, ':','LineWidth',2);
hold on

hold off

legend('legend1', 'legend2', 'legend3', 'legend4', 'FontSize', 12, 'Location','southeast')

title('Predictive Coding Accuracy', 'FontSize', 13)
xlabel('Epoch', 'FontSize', 13) 
ylabel('Loss', 'FontSize', 13) 

% Set plot 2 ranges
set(gca,'XTick',1:2:20);
axis([1 20 50 100])


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


