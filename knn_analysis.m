
root_dir='/om/group/evlab/Greta_Eghbal_manifolds/';
analyze_identifier=${run_analyze};
assignment=cellfun(@(x) strsplit(x,'='),strsplit(analyze_identifier,'-'),'uni',false);
cellfun(@(x) evalin('base',strcat(x{1},'=',"'",x{2},"'")),assignment(2:end),'uni',false);
model_identifier=${run_model};
file=${run_file};
layer=file([regexp(file,'layer'):(regexp(file,'_extracted'))-1]);
runKNN('root_dir',root_dir,'analyze_identifier',analyze_identifier,'model_identifier',model_identifier,'layer',layer,'dist_metric',dist_metric,'save_fig',true,'k',str2double(k),'num_subsamples',str2double(num_subsamples));
