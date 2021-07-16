function run_manifold_simulation_capacity(run_file)
load(erase(run_file,' '));
output_cell={};
for p=1:size(activation.projection_results,2)
    X = (activation.projection_results{p}.(activation.layer_name));
    XtotT={};
    for ii=1:size(X,1)
        X_class=double(squeeze(X(ii,:,1:size(X,3))));
        modif=0e-2*repmat(randn(size(X_class,1),1),1,size(X_class,2));
        XtotT{ii} = X_class;%+modif;\
    end
    options.n_rep =10;
    options.seed0 = 1;
    options.flag_NbyM =1;
    [output] = manifold_simcap_analysis(XtotT, options);
    output_cell=[output_cell;output];
end
save_id=strrep(erase(run_file," "),'extracted_v3.mat','capacity_v3.mat');
save(save_id,"output_cell");
end