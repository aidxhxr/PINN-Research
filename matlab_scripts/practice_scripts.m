%% MATLAB Tutorial
% Amirkhan Aidarkhan - January 2023

clear all
close all
clc

counter = 0
%
while counter >= 5;
    counter = counter - 1
end

%% Plotting
x = 0:0.1:5;
y = x.^2

plot(x, y, 'r+')
title('First graph')
xlabel('x-value')
ylabel('y-value')
grid on
hold
y2=x.^3;
y3=x.^4;
plot(x, y2, 'g*')
plot(x, y3)
hold off;
legend('Plot 1', 'Plot 2', 'Plot 3')

%% Subplotting 
subplot(311)
plot(x, y)
subplot(312)
plot(x, y2)
subplot(313)
plot(x, y3)