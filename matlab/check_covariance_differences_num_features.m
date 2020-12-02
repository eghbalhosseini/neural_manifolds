T=d3.ops_out.data;
T1=d5.ops_out.data;
sections=floor(linspace(10,size(T,2),50));
difference=[];
difference_std=[];
T1_idx=randi(size(T,1),5000,1);
T2_idx=randi(size(T,1),5000,1);
for k=sections
    disp(k);
    T_Cov=cov(T(T1_idx,1:k)');
    T1_Cov=cov(T1(T2_idx,1:k)');
    temp=T_Cov-T1_Cov;
    difference=[difference,mean(temp(:))];
    difference_std=[difference_std,std(temp(:))];
end 

figure;errorbar(sections,abs(difference),difference_std)
hold on ;
test=randn(5000,936);
test_cov=cov(test);
errorbar(sections(end),mean(test_cov(:)),std(test_cov(:)),'r')