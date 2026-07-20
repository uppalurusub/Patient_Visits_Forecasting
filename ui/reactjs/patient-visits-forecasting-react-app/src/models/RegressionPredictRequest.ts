/**
 * Regression Prediction Request
 */
export interface RegressionPredictRequest {

    model_name: string;

    /**
     * Total patient count
     */
    patient_count: number;

    /**
     * Total admissions
     */
    admissions: number;

    /**
     * Total discharges
     */
    discharges: number;

    /**
     * Bed occupancy percentage
     */
    occupancy: number;

    /**
     * Year
     */
    year: number;

    /**
     * Month (1-12)
     */
    month: number;

    /**
     * Day (1-31)
     */
    day: number;

    /**
     * Day of week (0-6)
     */
    dayofweek: number;

    /**
     * Quarter (1-4)
     */
    quarter: number;

    /**
     * Week number (1-53)
     */
    weekofyear: number;

}