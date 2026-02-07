import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import {
  ConflictPrediction,
  PredictionResponse,
  CountryRiskScore,
  ModelPerformance,
  MultiHorizonResponse,
  NewsStory,
  DashboardOverview
} from '../models/interfaces';

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  // v2 API endpoint
  private apiUrl = 'http://127.0.0.1:8080';

  constructor(private http: HttpClient) {}

  /**
   * Get latest predictions for all countries with optional filters
   */
  getLatestPredictions(filters?: {
    minProbability?: number;
    riskLevels?: string[];
    countries?: string[];
  }): Observable<ConflictPrediction[]> {
    let params = new HttpParams();

    if (filters?.minProbability) {
      params = params.set('min_probability', filters.minProbability.toString());
    }
    if (filters?.riskLevels?.length) {
      params = params.set('risk_levels', filters.riskLevels.join(','));
    }
    if (filters?.countries?.length) {
      params = params.set('countries', filters.countries.join(','));
    }

    return this.http.get<ConflictPrediction[]>(`${this.apiUrl}/api/predictions/latest`, { params });
  }

  /**
   * Get multi-horizon predictions (7, 14, 30 days) from v2 API
   */
  getMultiHorizonPredictions(filters?: {
    horizon?: number;
    minProbability?: number;
    riskLevels?: string[];
  }): Observable<MultiHorizonResponse> {
    let params = new HttpParams();

    if (filters?.horizon) {
      params = params.set('horizon', filters.horizon.toString());
    }
    if (filters?.minProbability) {
      params = params.set('min_probability', filters.minProbability.toString());
    }
    if (filters?.riskLevels?.length) {
      params = params.set('risk_levels', filters.riskLevels.join(','));
    }

    return this.http.get<MultiHorizonResponse>(`${this.apiUrl}/api/predictions/multi-horizon`, { params });
  }

  /**
   * Get detailed prediction for a specific country (v2 endpoint)
   */
  getCountryPrediction(country: string): Observable<PredictionResponse> {
    return this.http.get<PredictionResponse>(`${this.apiUrl}/api/predictions/country/${country}`);
  }

  /**
   * Get historical trend for a country (v2 - stored predictions)
   */
  getCountryTrend(country: string, days: number = 7, horizon: number = 7): Observable<any> {
    let params = new HttpParams()
      .set('days', days.toString())
      .set('horizon', horizon.toString());
    return this.http.get(`${this.apiUrl}/api/predictions/${country}/trend`, { params });
  }

  /**
   * Get risk scores for all countries
   */
  getRiskScores(limit?: number): Observable<CountryRiskScore[]> {
    let params = new HttpParams();
    if (limit) {
      params = params.set('limit', limit.toString());
    }
    return this.http.get<CountryRiskScore[]>(`${this.apiUrl}/api/risk-scores`, { params });
  }

  /**
   * Get model performance metrics (v2 - calibration data)
   */
  getModelPerformance(): Observable<ModelPerformance> {
    return this.http.get<ModelPerformance>(`${this.apiUrl}/api/model-performance`);
  }

  /**
   * Get dashboard overview with predictions and stats
   */
  getDashboardOverview(): Observable<any> {
    return this.http.get(`${this.apiUrl}/api/dashboard/overview`);
  }

  /**
   * Get positive news stories (cooperation events)
   */
  getPositiveNews(country?: string): Observable<any[]> {
    let params = new HttpParams();
    if (country) {
      params = params.set('country', country);
    }
    return this.http.get<any[]>(`${this.apiUrl}/api/news/positive`, { params });
  }

  /**
   * Get negative news stories (conflict events)
   */
  getNegativeNews(country?: string): Observable<any[]> {
    let params = new HttpParams();
    if (country) {
      params = params.set('country', country);
    }
    return this.http.get<any[]>(`${this.apiUrl}/api/news/negative`, { params });
  }

  /**
   * Get data status (prediction counts, date range)
   */
  getDataStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/api/data/status`);
  }

  /**
   * Trigger new prediction generation
   */
  collectMissingData(): Observable<any> {
    return this.http.post(`${this.apiUrl}/api/data/collect`, {});
  }

  /**
   * Get model status (horizon-specific models)
   */
  getModelStatus(): Observable<any> {
    return this.http.get(`${this.apiUrl}/api/models/status`);
  }

  /**
   * Get calibration metrics for a horizon
   */
  getCalibration(horizon: number): Observable<any> {
    return this.http.get(`${this.apiUrl}/api/calibration/${horizon}`);
  }

  /**
   * Get list of countries with predictions
   */
  getCountries(): Observable<any> {
    return this.http.get(`${this.apiUrl}/api/countries`);
  }

  /**
   * Get live GDELT events with coordinates for scatter map
   */
  getLiveEvents(limit: number = 500): Observable<any> {
    let params = new HttpParams().set('limit', limit.toString());
    return this.http.get(`${this.apiUrl}/api/events/live`, { params });
  }

  /**
   * Health check
   */
  healthCheck(): Observable<any> {
    return this.http.get(`${this.apiUrl}/`);
  }
}
