function [center,out]=km(k,data)
[m,n]=size(data);
out=zeros(m,n+1);
out(:,1:n)=data;
center=zeros(k,n);
for i=1:k
center(i,:)=data(randi(m,1),:);
end
ncent=zeros(k,n);
dist=zeros(1,k);
for i=1:m
    for j=1:k
        dist(:,j)=distance(data(i,:),center(j,:));
        [~,temp]=min(dist);
        out(i,n+1)=temp;
    end
end
count=0;
for i=1:k
    for j=1:m
        if out(j,n+1)==i
    ncent(i,:)=ncent(i,:)+data(j,:);
    count=count+1;
        end
    end
    ncent(i,:)=ncent(i,:)/count;
    count=0;
end
if distance(ncent,center)>=0.000001
    center=ncent;
    ncent=zeros(k,n);
    dist=zeros(1,k);
for i=1:m
    for j=1:k
        dist(:,j)=distance(data(i,:),center(j,:));
        [~,temp]=min(dist);
        out(i,n+1)=temp;
    end
end
    for i=1:k
    for j=1:m
        if out(j,n+1)==i
    ncent(i,:)=ncent(i,:)+data(j,:);
    count=count+1;
        end
    end
    ncent(i,:)=ncent(i,:)/count;
    count=0;
    end
else
    center=ncent;
end

end
function [d]=distance(a,b)
d=norm(a-b);
end
