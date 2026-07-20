/**
 * Number Formatting Utilities
 */

/**
 * Format number with commas
 */
export const formatNumber = (
    value: number,
    decimals = 0
): string => {

    return new Intl.NumberFormat("en-IN", {

        minimumFractionDigits: decimals,

        maximumFractionDigits: decimals

    }).format(value);

};

/**
 * Format decimal number
 */
export const formatDecimal = (
    value: number,
    decimals = 2
): string => {

    return value.toFixed(decimals);

};

/**
 * Format percentage
 */
export const formatPercentage = (
    value: number,
    decimals = 2
): string => {

    return `${value.toFixed(decimals)}%`;

};

/**
 * Format Indian Currency
 */
export const formatCurrency = (
    value: number,
    currency = "INR"
): string => {

    return new Intl.NumberFormat("en-IN", {

        style: "currency",

        currency

    }).format(value);

};

/**
 * Format US Currency
 */
export const formatUSCurrency = (
    value: number
): string => {

    return new Intl.NumberFormat("en-US", {

        style: "currency",

        currency: "USD"

    }).format(value);

};

/**
 * Format compact numbers
 *
 * 1250 -> 1.25K
 * 1200000 -> 1.2M
 */
export const formatCompact = (
    value: number
): string => {

    return new Intl.NumberFormat("en", {

        notation: "compact",

        maximumFractionDigits: 2

    }).format(value);

};

/**
 * Format forecast value
 */
export const formatForecast = (
    value: number
): string => {

    return value.toFixed(2);

};

/**
 * Format nullable values
 */
export const formatNullable = (
    value?: number | null,
    fallback = "-"
): string => {

    if (value === undefined || value === null) {

        return fallback;

    }

    return formatNumber(value, 2);

};

/**
 * Format millions
 *
 * 12500000 -> ₹12.50 M
 */
export const formatMillions = (
    value: number
): string => {

    return `₹${(value / 1_000_000).toFixed(2)} M`;

};

/**
 * Format billions
 */
export const formatBillions = (
    value: number
): string => {

    return `₹${(value / 1_000_000_000).toFixed(2)} B`;

};