import * as tf from '@tensorflow/tfjs';

export type DataBatch = {
	xs: tf.Tensor2D;
	ys: tf.Tensor2D;
};

export type DataSourceConfig = {
	imagesSpriteURL: string;
	labelsURL: string;
	imageSize: number;
	numClasses: number;
	numDatasetElements: number;
	numTrainElements: number;
};

export interface DataSource {
	nextTrainBatch: (batchSize: number) => DataBatch;
	nextTestBatch: (batchSize: number) => DataBatch;
	loadData: () => Promise<void>;
}
