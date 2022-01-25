.PHONY: visualize_ag_reldn
visualize_ag_reldn:
	python tools/demo/demo_image.py \
		--config_file sgg_configs/ag_vrd/rel_danfeiX_FPN50_reldn.yaml \
		--img_file demo/ag_demo.png \
		--save_file ./results/ag/reldn/ag_demo.png \
		--visualize_relation DATASETS.LABELMAP_FILE "actiongenome/labelmap.json"
