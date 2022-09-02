%自己产生高斯数据%
%第一组数据
mu1=[0 0 ];  %均值
S1=[0.1 0 ;0 0.1];  %协方差
data1=mvnrnd(mu1,S1,100);%产生高斯分布数据
%第二组数据
mu2=[1.5 1.5];
S2=[0.1 0 ;0 0.1];
data2=mvnrnd(mu2,S2,100);
% 第三组数据
mu3=[-1.5 1.5];
S3=[1 0 ;0 0.1];
data3=mvnrnd(mu3,S3,100);
data=[data1;data2;data3];
csvwrite('gaussdata.csv',data);
figure(1);
hold on;
plot(data1(:,1),data1(:,2),'b*')
plot(data2(:,1),data2(:,2),'r*')
plot(data3(:,1),data3(:,2),'g*')
title('gauss data')


% [m,n]=size(data);
% [catagory_cent,catagory]=km(3,data);
% catagory_cent
% figure(2);
% hold on;
% for i=1:m
%     if catagory(i,n+1)==1
%         plot(catagory(i,1),catagory(i,2),'b*')
%     elseif catagory(i,n+1)==2
%             plot(catagory(i,1),catagory(i,2),'r*')
%     else
%             plot(catagory(i,1),catagory(i,2),'g*')
%     end
% end 
% title('result of k-means')



