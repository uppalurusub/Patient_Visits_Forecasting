export interface PredictionRequest{
 model_name:string;
 horizon:number;
}
export interface RegressionPredictionRequest{
 [key:string]:string|number|boolean;
}
