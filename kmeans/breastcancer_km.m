%采用UCI数据集
download_data1=csvread('dataR2.csv',1,1,[1,1,116,8]);
real=csvread('dataR2.csv',1,9,[1,9,116,9]);
[cent1,result1]=km(2,download_data1);
rightness=1-sum(abs(result1(:,9)-real))/116
[real,result1(:,9)]'

% download_data=csvread('c1_raw.csv',1,0,[1,0,1111,17]);
% state=csvread('c1_raw.csv',1,18,[1,18,1111,18]);
% real_state=ones(1111,1);
% [cent,result]=km(5,download_data);
% for i=1:1111
%     if state(i)=='Rest'
%         real_state(i)=1;
%     elseif state(i)=='Preparation'
%         real_state(i)=2;
%     elseif state(i)=='Stroke'
%         real_state(i)=3;
%     elseif state(i)=='Hold'
%         real_state(i)=4;
%     else
%         state(i)=='Retraction'
%         real_state(i)=5;
%     end
% end
%rightness=1-sum(abs(result1(:,9)-real))/116