function knn_analysis(file,model_identifier,analyze_identifier)
root_dir='/nese/mit/group/evlab/projects/Greta_Eghbal_manifolds/';
assignment=cellfun(@(x) strsplit(x,'='),strsplit(analyze_identifier,'-'),'uni',false);
cellfun(@(x) evalin('base',strcat(x{1},'=',"'",x{2},"'")),assignment(2:end),'uni',false);
layer=file([regexp(file,'layer'):(regexp(file,'_extracted'))-1]);
print("")
runKNN('root_dir',root_dir,'analyze_identifier',analyze_identifier,'model_identifier',model_identifier,'layer',layer,'dist_metric',dist_metric,'save_fig',true,'k',str2double(k),'num_subsamples',str2double(num_subsamples));
