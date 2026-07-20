export interface PredictionResponse{
 success:boolean;
 message:string;
 model_name:string;
 prediction:any;
}
export interface RegressionPredictionResponse{
 success:boolean;
 prediction:any;
}
