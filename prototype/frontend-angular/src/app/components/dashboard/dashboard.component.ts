import { Component, OnInit, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatTableModule } from '@angular/material/table';
import { MatChipsModule } from '@angular/material/chips';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatTabsModule } from '@angular/material/tabs';
import { MatTooltipModule } from '@angular/material/tooltip';
import { ApiService } from '../../services/api.service';
import { ConflictPrediction, ModelPerformance, CountryRiskScore, MultiHorizonResponse } from '../../models/interfaces';
import * as Plotly from 'plotly.js-geo-dist';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    FormsModule,
    MatCardModule,
    MatTableModule,
    MatChipsModule,
    MatProgressSpinnerModule,
    MatFormFieldModule,
    MatSelectModule,
    MatSliderModule,
    MatButtonModule,
    MatIconModule,
    MatTabsModule,
    MatTooltipModule
  ],
  templateUrl: './dashboard.component.html',
  styleUrls: ['./dashboard.component.scss']
})
export class DashboardComponent implements OnInit, AfterViewInit {
  predictions: ConflictPrediction[] = [];
  filteredPredictions: ConflictPrediction[] = [];
  riskScores: CountryRiskScore[] = [];
  performance: ModelPerformance | null = null;
  loading = true;
  error: string | null = null;

  // News data
  positiveNews: any[] = [];
  negativeNews: any[] = [];

  // Dashboard overview
  dashboardData: any = null;

  // Data status
  dataStatus: any = null;
  isCollectingData = false;
  collectionMessage: string | null = null;

  // Multi-horizon data
  multiHorizonData: MultiHorizonResponse | null = null;
  selectedHorizon: number = 7;
  selectedHorizonIndex: number = 0;
  availableHorizons: number[] = [7, 14, 30];

  // Map view mode
  mapViewMode: 'choropleth' | 'scatter' = 'choropleth';

  // Live GDELT events for scatter map
  liveEvents: any[] = [];
  liveEventsLoading = false;

  // Filters
  selectedRiskLevels: string[] = [];
  selectedCountries: string[] = [];
  minProbability: number = 0;

  // Available options
  riskLevels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'];
  allCountries: string[] = [];

  // Table columns
  displayedColumns: string[] = ['country', 'target_date', 'probability', 'risk_level', 'confidence'];

  constructor(private apiService: ApiService) {}

  ngOnInit(): void {
    this.loadData();
    this.loadDataStatus();
  }

  ngAfterViewInit(): void {
    // Charts will be rendered after data loads
  }

  loadDataStatus(): void {
    this.apiService.getDataStatus().subscribe({
      next: (status) => {
        this.dataStatus = status;
      },
      error: (err) => {
        console.error('Error loading data status:', err);
      }
    });
  }

  collectMissingData(): void {
    this.isCollectingData = true;
    this.collectionMessage = null;
    
    this.apiService.collectMissingData().subscribe({
      next: (result) => {
        this.collectionMessage = result.message;
        this.isCollectingData = false;
        // Refresh data status
        this.loadDataStatus();
      },
      error: (err) => {
        this.collectionMessage = 'Failed to trigger data collection';
        this.isCollectingData = false;
        console.error('Error collecting data:', err);
      }
    });
  }

  loadData(): void {
    this.loading = true;
    this.error = null;

    // Load dashboard overview (includes predictions and news)
    this.apiService.getDashboardOverview().subscribe({
      next: (data) => {
        console.log('Dashboard data received:', data);
        this.dashboardData = data;
        this.predictions = data.predictions || [];
        this.filteredPredictions = this.predictions;
        this.positiveNews = data.positive_news || [];
        this.negativeNews = data.negative_news || [];
        
        // Set up countries list
        this.allCountries = [...new Set(this.predictions.map(p => p.country))].sort();
        
        this.loading = false;
        this.renderCharts();
      },
      error: (err) => {
        this.error = 'Failed to load dashboard data. Make sure the backend is running on http://127.0.0.1:8080';
        this.loading = false;
        console.error('Error loading dashboard:', err);
      }
    });

    // Load multi-horizon predictions
    this.apiService.getMultiHorizonPredictions().subscribe({
      next: (data) => {
        this.multiHorizonData = data;
        this.availableHorizons = data.horizons || [7, 14, 30];
        this.selectedHorizon = this.availableHorizons[0] || 7;
        this.updatePredictionsForHorizon();
      },
      error: (err) => {
        console.error('Error loading multi-horizon:', err);
        // Set defaults
        this.availableHorizons = [7, 14, 30];
        this.selectedHorizon = 7;
      }
    });

    // Load performance
    this.apiService.getModelPerformance().subscribe({
      next: (data) => {
        this.performance = data;
        this.renderPerformanceChart();
      },
      error: (err) => {
        console.error('Error loading performance:', err);
      }
    });

    // Load risk scores
    this.apiService.getRiskScores(10).subscribe({
      next: (data) => {
        this.riskScores = data;
        this.renderRiskScoresChart();
      },
      error: (err) => {
        console.error('Error loading risk scores:', err);
      }
    });
  }

  updatePredictionsForHorizon(): void {
    if (!this.multiHorizonData) return;

    const horizonKey = this.selectedHorizon.toString();
    this.predictions = this.multiHorizonData.predictions_by_horizon[horizonKey] || [];
    this.allCountries = [...new Set(this.predictions.map(p => p.country))].sort();
    this.applyFilters();
  }

  onHorizonChange(index: number): void {
    this.selectedHorizonIndex = index;
    this.selectedHorizon = this.availableHorizons[index];
    this.updatePredictionsForHorizon();
    this.renderCharts();
  }

  applyFilters(): void {
    this.filteredPredictions = this.predictions.filter(pred => {
      // Risk level filter
      if (this.selectedRiskLevels.length > 0 && !this.selectedRiskLevels.includes(pred.risk_level)) {
        return false;
      }

      // Country filter
      if (this.selectedCountries.length > 0 && !this.selectedCountries.includes(pred.country)) {
        return false;
      }

      // Probability filter
      if (pred.conflict_probability < this.minProbability / 100) {
        return false;
      }

      return true;
    });

    this.renderCharts();
  }

  clearFilters(): void {
    this.selectedRiskLevels = [];
    this.selectedCountries = [];
    this.minProbability = 0;
    this.applyFilters();
  }

  // Quick stat helper methods
  getCriticalCount(): number {
    return this.predictions.filter(p => p.risk_level === 'CRITICAL').length;
  }

  getHighCount(): number {
    return this.predictions.filter(p => p.risk_level === 'HIGH').length;
  }

  getMediumCount(): number {
    return this.predictions.filter(p => p.risk_level === 'MEDIUM').length;
  }

  getLowCount(): number {
    return this.predictions.filter(p => p.risk_level === 'LOW').length;
  }

  setMapView(mode: 'choropleth' | 'scatter'): void {
    this.mapViewMode = mode;
    if (mode === 'scatter' && this.liveEvents.length === 0) {
      this.loadLiveEvents();
    } else {
      this.renderGlobalMap();
    }
  }

  loadLiveEvents(): void {
    this.liveEventsLoading = true;
    this.apiService.getLiveEvents(500).subscribe({
      next: (response) => {
        this.liveEvents = response.events || [];
        this.liveEventsLoading = false;
        this.renderGlobalMap();
      },
      error: (err) => {
        console.error('Error loading live events:', err);
        this.liveEventsLoading = false;
        // Fallback to country-level scatter if live events fail
        this.renderScatterMapFallback();
      }
    });
  }

  renderCharts(): void {
    setTimeout(() => {
      this.renderGlobalMap();
      this.renderRiskDonut();
      this.renderTopRiskBar();
      this.renderRegionalBar();
    }, 100);
  }

  renderGlobalMap(): void {
    if (this.mapViewMode === 'choropleth') {
      this.renderGlobalHeatmap();
    } else {
      this.renderScatterMap();
    }
  }

  renderGlobalHeatmap(): void {
    // Use shared country mapping
    const codeToCountry = this.getCodeToCountryMap();

    const locations: string[] = [];
    const zValues: number[] = [];
    const hoverTexts: string[] = [];

    // The API already returns ISO-3 codes in the country field
    this.filteredPredictions.forEach(pred => {
      // Use the country code directly since API returns ISO-3 codes
      const countryCode = pred.country;
      const countryName = codeToCountry[countryCode] || countryCode;
      
      locations.push(countryCode);
      zValues.push(pred.conflict_probability * 100);
      hoverTexts.push(
        `<b>${countryName}</b><br>` +
        `Probability: ${(pred.conflict_probability * 100).toFixed(1)}%<br>` +
        `Risk Level: ${pred.risk_level}<br>` +
        `Confidence: ${(pred.confidence * 100).toFixed(1)}%`
      );
    });

    const trace: any = {
      type: 'choropleth',
      locations: locations,
      z: zValues,
      text: hoverTexts,
      hoverinfo: 'text',
      colorscale: [
        [0, '#4caf50'],      // Low - green
        [0.25, '#8bc34a'],   // Low-medium
        [0.5, '#ffeb3b'],    // Medium - yellow
        [0.75, '#ff9800'],   // High - orange
        [1, '#f44336']       // Critical - red
      ],
      colorbar: {
        title: 'Conflict<br>Probability (%)',
        ticksuffix: '%',
        thickness: 15,
        len: 0.5
      },
      marker: {
        line: {
          color: 'rgba(0,0,0,0.2)',
          width: 0.5
        }
      }
    };

    const layout: any = {
      geo: {
        showframe: false,
        showcoastlines: true,
        coastlinecolor: 'rgba(0,0,0,0.3)',
        projection: {
          type: 'natural earth'
        },
        showland: true,
        landcolor: '#f0f0f0',
        showocean: true,
        oceancolor: '#e6f2ff',
        showcountries: true,
        countrycolor: 'rgba(0,0,0,0.1)'
      },
      margin: { t: 10, b: 10, l: 10, r: 10 },
      paper_bgcolor: 'transparent',
      height: 450
    };

    const config: any = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      displaylogo: false
    };

    Plotly.newPlot('global-heatmap', [trace], layout, config);
  }

  renderScatterMap(): void {
    // Use live GDELT events with actual lat/long coordinates
    if (this.liveEvents.length === 0) {
      // Show loading or fallback to country-level
      this.renderScatterMapFallback();
      return;
    }

    const lats: number[] = [];
    const lons: number[] = [];
    const sizes: number[] = [];
    const colors: string[] = [];
    const hoverTexts: string[] = [];

    this.liveEvents.forEach(event => {
      lats.push(event.lat);
      lons.push(event.lon);
      // Size based on mentions (importance) - scale between 5 and 25
      const baseSize = Math.min(25, Math.max(5, 5 + (event.num_mentions / 10)));
      sizes.push(baseSize);
      colors.push(this.getRiskLevelColor(event.risk_level));
      
      hoverTexts.push(
        `<b>${event.location || 'Unknown Location'}</b><br>` +
        `Actors: ${event.actor1}  ${event.actor2}<br>` +
        `Goldstein Scale: ${event.goldstein_scale.toFixed(1)}<br>` +
        `Risk Level: ${event.risk_level}<br>` +
        `Mentions: ${event.num_mentions}<br>` +
        `Tone: ${event.avg_tone.toFixed(1)}`
      );
    });

    const trace: any = {
      type: 'scattergeo',
      lat: lats,
      lon: lons,
      text: hoverTexts,
      hoverinfo: 'text',
      marker: {
        size: sizes,
        color: colors,
        opacity: 0.7,
        line: {
          color: 'rgba(255,255,255,0.5)',
          width: 0.5
        }
      }
    };

    const layout: any = {
      geo: {
        showframe: false,
        showcoastlines: true,
        coastlinecolor: 'rgba(0,0,0,0.3)',
        projection: { type: 'natural earth' },
        showland: true,
        landcolor: '#e8e8e8',
        showocean: true,
        oceancolor: '#e6f2ff',
        showcountries: true,
        countrycolor: 'rgba(0,0,0,0.2)'
      },
      margin: { t: 10, b: 10, l: 10, r: 10 },
      paper_bgcolor: 'transparent',
      height: 450,
      annotations: [{
        text: `${this.liveEvents.length} live GDELT events`,
        showarrow: false,
        x: 0.01,
        y: 0.01,
        xref: 'paper',
        yref: 'paper',
        font: { size: 11, color: '#666' }
      }]
    };

    const config: any = {
      responsive: true,
      displayModeBar: true,
      modeBarButtonsToRemove: ['lasso2d', 'select2d'],
      displaylogo: false
    };

    Plotly.newPlot('global-heatmap', [trace], layout, config);
  }

  renderScatterMapFallback(): void {
    // Fallback to country-level scatter when live events aren't available
    const countryCoords: { [key: string]: { lat: number, lon: number } } = {
      'AFG': { lat: 33.93, lon: 67.71 }, 'ALB': { lat: 41.15, lon: 20.17 },
      'DZA': { lat: 28.03, lon: 1.66 }, 'ARG': { lat: -38.42, lon: -63.62 },
      'AUS': { lat: -25.27, lon: 133.78 }, 'AUT': { lat: 47.52, lon: 14.55 },
      'BGD': { lat: 23.68, lon: 90.36 }, 'BEL': { lat: 50.50, lon: 4.47 },
      'BRA': { lat: -14.24, lon: -51.93 }, 'BGR': { lat: 42.73, lon: 25.49 },
      'CAN': { lat: 56.13, lon: -106.35 }, 'CHL': { lat: -35.68, lon: -71.54 },
      'CHN': { lat: 35.86, lon: 104.20 }, 'COL': { lat: 4.57, lon: -74.30 },
      'CZE': { lat: 49.82, lon: 15.47 }, 'DNK': { lat: 56.26, lon: 9.50 },
      'EGY': { lat: 26.82, lon: 30.80 }, 'ETH': { lat: 9.15, lon: 40.49 },
      'FIN': { lat: 61.92, lon: 25.75 }, 'FRA': { lat: 46.23, lon: 2.21 },
      'DEU': { lat: 51.17, lon: 10.45 }, 'GHA': { lat: 7.95, lon: -1.02 },
      'GRC': { lat: 39.07, lon: 21.82 }, 'HUN': { lat: 47.16, lon: 19.50 },
      'IND': { lat: 20.59, lon: 78.96 }, 'IDN': { lat: -0.79, lon: 113.92 },
      'IRN': { lat: 32.43, lon: 53.69 }, 'IRQ': { lat: 33.22, lon: 43.68 },
      'IRL': { lat: 53.14, lon: -7.69 }, 'ISR': { lat: 31.05, lon: 34.85 },
      'ITA': { lat: 41.87, lon: 12.57 }, 'JPN': { lat: 36.20, lon: 138.25 },
      'USA': { lat: 37.09, lon: -95.71 }, 'GBR': { lat: 55.38, lon: -3.44 },
      'UKR': { lat: 48.38, lon: 31.17 }, 'RUS': { lat: 61.52, lon: 105.32 },
      'SYR': { lat: 34.80, lon: 39.00 }, 'YEM': { lat: 15.55, lon: 48.52 }
    };

    const lats: number[] = [];
    const lons: number[] = [];
    const sizes: number[] = [];
    const colors: string[] = [];
    const hoverTexts: string[] = [];
    const codeToCountry = this.getCodeToCountryMap();

    this.filteredPredictions.forEach(pred => {
      const coords = countryCoords[pred.country];
      if (coords) {
        lats.push(coords.lat);
        lons.push(coords.lon);
        sizes.push(10 + pred.conflict_probability * 30);
        colors.push(this.getRiskLevelColor(pred.risk_level));
        const countryName = codeToCountry[pred.country] || pred.country;
        hoverTexts.push(
          `<b>${countryName}</b><br>` +
          `Probability: ${(pred.conflict_probability * 100).toFixed(1)}%<br>` +
          `Risk Level: ${pred.risk_level}`
        );
      }
    });

    const trace: any = {
      type: 'scattergeo',
      lat: lats,
      lon: lons,
      text: hoverTexts,
      hoverinfo: 'text',
      marker: {
        size: sizes,
        color: colors,
        opacity: 0.8,
        line: { color: 'white', width: 1 }
      }
    };

    const layout: any = {
      geo: {
        showframe: false,
        showcoastlines: true,
        coastlinecolor: 'rgba(0,0,0,0.3)',
        projection: { type: 'natural earth' },
        showland: true,
        landcolor: '#e8e8e8',
        showocean: true,
        oceancolor: '#e6f2ff',
        showcountries: true,
        countrycolor: 'rgba(0,0,0,0.2)'
      },
      margin: { t: 10, b: 10, l: 10, r: 10 },
      paper_bgcolor: 'transparent',
      height: 450,
      annotations: [{
        text: 'Country-level view (live feed unavailable)',
        showarrow: false,
        x: 0.01,
        y: 0.01,
        xref: 'paper',
        yref: 'paper',
        font: { size: 11, color: '#999' }
      }]
    };

    Plotly.newPlot('global-heatmap', [trace], layout, { responsive: true, displayModeBar: false });
  }

  getCodeToCountryMap(): { [key: string]: string } {
    return {
      'AFG': 'Afghanistan', 'ALB': 'Albania', 'DZA': 'Algeria', 'ARG': 'Argentina',
      'AUS': 'Australia', 'AUT': 'Austria', 'BGD': 'Bangladesh', 'BEL': 'Belgium',
      'BRA': 'Brazil', 'BGR': 'Bulgaria', 'KHM': 'Cambodia', 'CMR': 'Cameroon',
      'CAN': 'Canada', 'CHL': 'Chile', 'CHN': 'China', 'COL': 'Colombia',
      'COG': 'Congo', 'COD': 'DR Congo', 'HRV': 'Croatia', 'CUB': 'Cuba',
      'CZE': 'Czech Republic', 'DNK': 'Denmark', 'EGY': 'Egypt', 'ETH': 'Ethiopia',
      'FIN': 'Finland', 'FRA': 'France', 'DEU': 'Germany', 'GHA': 'Ghana',
      'GRC': 'Greece', 'HUN': 'Hungary', 'IND': 'India', 'IDN': 'Indonesia',
      'IRN': 'Iran', 'IRQ': 'Iraq', 'IRL': 'Ireland', 'ISR': 'Israel',
      'ITA': 'Italy', 'JPN': 'Japan', 'JOR': 'Jordan', 'KAZ': 'Kazakhstan',
      'KEN': 'Kenya', 'KWT': 'Kuwait', 'LBN': 'Lebanon', 'LBY': 'Libya',
      'MYS': 'Malaysia', 'MEX': 'Mexico', 'MAR': 'Morocco', 'MMR': 'Myanmar',
      'NLD': 'Netherlands', 'NZL': 'New Zealand', 'NGA': 'Nigeria', 'PRK': 'North Korea',
      'NOR': 'Norway', 'PAK': 'Pakistan', 'PSE': 'Palestine', 'PER': 'Peru',
      'PHL': 'Philippines', 'POL': 'Poland', 'PRT': 'Portugal', 'QAT': 'Qatar',
      'ROU': 'Romania', 'RUS': 'Russia', 'SAU': 'Saudi Arabia', 'SRB': 'Serbia',
      'SGP': 'Singapore', 'SOM': 'Somalia', 'ZAF': 'South Africa', 'KOR': 'South Korea',
      'ESP': 'Spain', 'LKA': 'Sri Lanka', 'SDN': 'Sudan', 'SWE': 'Sweden',
      'CHE': 'Switzerland', 'SYR': 'Syria', 'TWN': 'Taiwan', 'THA': 'Thailand',
      'TUN': 'Tunisia', 'TUR': 'Turkey', 'UKR': 'Ukraine', 'ARE': 'UAE',
      'GBR': 'United Kingdom', 'USA': 'United States', 'VEN': 'Venezuela',
      'VNM': 'Vietnam', 'YEM': 'Yemen', 'ZWE': 'Zimbabwe', 'CAF': 'Central African Rep.',
      'VCT': 'St. Vincent', 'MRT': 'Mauritania', 'TGO': 'Togo', 'PRY': 'Paraguay',
      'GRD': 'Grenada', 'ATG': 'Antigua', 'BWA': 'Botswana', 'MLT': 'Malta',
      'MKD': 'North Macedonia', 'SLE': 'Sierra Leone', 'MDA': 'Moldova', 'NPL': 'Nepal',
      'ZMB': 'Zambia', 'LBR': 'Liberia', 'CPV': 'Cape Verde', 'LVA': 'Latvia',
      'ARM': 'Armenia', 'SVK': 'Slovakia', 'GAB': 'Gabon', 'SUR': 'Suriname',
      'CIV': 'Ivory Coast', 'MNG': 'Mongolia', 'LAO': 'Laos', 'CRI': 'Costa Rica',
      'PNG': 'Papua New Guinea', 'URY': 'Uruguay', 'FJI': 'Fiji', 'TKM': 'Turkmenistan',
      'AGO': 'Angola', 'EST': 'Estonia', 'LUX': 'Luxembourg', 'BTN': 'Bhutan',
      'KGZ': 'Kyrgyzstan', 'BHR': 'Bahrain', 'AZE': 'Azerbaijan', 'UZB': 'Uzbekistan',
      'OMN': 'Oman', 'SSD': 'South Sudan', 'TJK': 'Tajikistan', 'GNB': 'Guinea-Bissau',
      'GIN': 'Guinea', 'BEN': 'Benin', 'MWI': 'Malawi', 'RWA': 'Rwanda', 'BDI': 'Burundi',
      'ERI': 'Eritrea', 'DJI': 'Djibouti', 'SWZ': 'Eswatini', 'LSO': 'Lesotho',
      'GEO': 'Georgia', 'BLR': 'Belarus', 'LTU': 'Lithuania', 'SVN': 'Slovenia',
      'BIH': 'Bosnia', 'MNE': 'Montenegro', 'XKX': 'Kosovo',
      'CYP': 'Cyprus', 'ISL': 'Iceland', 'HTI': 'Haiti', 'DOM': 'Dominican Rep.',
      'JAM': 'Jamaica', 'TTO': 'Trinidad', 'BHS': 'Bahamas', 'GUY': 'Guyana',
      'SLV': 'El Salvador', 'GTM': 'Guatemala', 'HND': 'Honduras', 'NIC': 'Nicaragua',
      'PAN': 'Panama', 'ECU': 'Ecuador', 'BOL': 'Bolivia', 'NER': 'Niger',
      'TCD': 'Chad', 'MLI': 'Mali', 'BFA': 'Burkina Faso', 'SEN': 'Senegal',
      'GMB': 'Gambia', 'MUS': 'Mauritius', 'MDG': 'Madagascar', 'MOZ': 'Mozambique',
      'NAM': 'Namibia', 'TZA': 'Tanzania', 'UGA': 'Uganda'
    };
  }

  renderRiskDonut(): void {
    const riskCounts: any = { 'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0 };
    this.filteredPredictions.forEach(p => riskCounts[p.risk_level]++);

    const trace: any = {
      labels: ['Critical', 'High', 'Medium', 'Low'],
      values: [riskCounts['CRITICAL'], riskCounts['HIGH'], riskCounts['MEDIUM'], riskCounts['LOW']],
      type: 'pie',
      hole: 0.5,
      marker: {
        colors: ['#f44336', '#ff9800', '#ffc107', '#4caf50']
      },
      textinfo: 'label+value',
      textposition: 'outside',
      hovertemplate: '<b>%{label}</b><br>%{value} countries<br>%{percent}<extra></extra>'
    };

    const layout: any = {
      showlegend: false,
      margin: { t: 30, b: 30, l: 30, r: 30 },
      paper_bgcolor: 'transparent',
      height: 280,
      annotations: [{
        text: `${this.filteredPredictions.length}<br>Countries`,
        showarrow: false,
        font: { size: 16, color: '#333' }
      }]
    };

    Plotly.newPlot('risk-donut', [trace], layout, { responsive: true, displayModeBar: false });
  }

  renderTopRiskBar(): void {
    const top10 = [...this.filteredPredictions]
      .sort((a, b) => b.conflict_probability - a.conflict_probability)
      .slice(0, 10)
      .reverse();  // Reverse for horizontal bar display

    const codeToCountry = this.getCodeToCountryMap();

    const trace: any = {
      y: top10.map(p => codeToCountry[p.country] || p.country),
      x: top10.map(p => p.conflict_probability * 100),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: top10.map(p => this.getRiskLevelColor(p.risk_level)),
        line: { color: 'white', width: 1 }
      },
      text: top10.map(p => `${(p.conflict_probability * 100).toFixed(1)}%`),
      textposition: 'outside',
      hovertemplate: '<b>%{y}</b><br>Probability: %{x:.1f}%<extra></extra>'
    };

    const layout: any = {
      margin: { t: 20, b: 40, l: 120, r: 60 },
      xaxis: { 
        title: 'Conflict Probability (%)',
        range: [0, 100],
        ticksuffix: '%'
      },
      yaxis: { automargin: true },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      height: 280
    };

    Plotly.newPlot('top-risk-bar', [trace], layout, { responsive: true, displayModeBar: false });
  }

  renderRegionalBar(): void {
    // Regional groupings based on country codes
    const regions: { [key: string]: string[] } = {
      'Middle East': ['IRQ', 'SYR', 'YEM', 'LBN', 'JOR', 'ISR', 'PSE', 'SAU', 'ARE', 'KWT', 'QAT', 'OMN', 'BHR', 'IRN'],
      'Africa': ['EGY', 'LBY', 'SDN', 'SSD', 'ETH', 'SOM', 'KEN', 'NGA', 'GHA', 'ZAF', 'COD', 'CAF', 'MLI', 'NER', 'TCD'],
      'Europe': ['UKR', 'RUS', 'POL', 'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'GRC', 'TUR', 'SRB', 'ROU', 'BGR', 'HUN'],
      'Asia': ['CHN', 'IND', 'PAK', 'BGD', 'MMR', 'THA', 'VNM', 'IDN', 'PHL', 'JPN', 'KOR', 'MYS', 'SGP', 'AFG'],
      'Americas': ['USA', 'MEX', 'BRA', 'ARG', 'COL', 'VEN', 'PER', 'CHL', 'CAN', 'CUB', 'HTI', 'GTM', 'SLV', 'HND']
    };

    const regionAvg: { name: string, avg: number, count: number }[] = [];

    Object.entries(regions).forEach(([region, codes]) => {
      const regionPreds = this.filteredPredictions.filter(p => codes.includes(p.country));
      if (regionPreds.length > 0) {
        const avg = regionPreds.reduce((sum, p) => sum + p.conflict_probability, 0) / regionPreds.length;
        regionAvg.push({ name: region, avg: avg * 100, count: regionPreds.length });
      }
    });

    regionAvg.sort((a, b) => b.avg - a.avg);

    const trace: any = {
      x: regionAvg.map(r => r.name),
      y: regionAvg.map(r => r.avg),
      type: 'bar',
      marker: {
        color: regionAvg.map(r => {
          if (r.avg >= 50) return '#f44336';
          if (r.avg >= 35) return '#ff9800';
          if (r.avg >= 20) return '#ffc107';
          return '#4caf50';
        })
      },
      text: regionAvg.map(r => `${r.avg.toFixed(1)}%`),
      textposition: 'outside',
      hovertemplate: '<b>%{x}</b><br>Avg Risk: %{y:.1f}%<br>Countries: ' + 
        regionAvg.map(r => r.count).join(', ') + '<extra></extra>'
    };

    const layout: any = {
      margin: { t: 20, b: 80, l: 50, r: 20 },
      yaxis: { 
        title: 'Avg Risk %',
        range: [0, Math.max(...regionAvg.map(r => r.avg)) * 1.2]
      },
      xaxis: { tickangle: -30 },
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      height: 280
    };

    Plotly.newPlot('regional-bar', [trace], layout, { responsive: true, displayModeBar: false });
  }

  renderProbabilityDistribution(): void {
    const probabilities = this.filteredPredictions.map(p => p.conflict_probability * 100);

    const trace: any = {
      x: probabilities,
      type: 'histogram',
      nbinsx: 20,
      marker: {
        color: '#1976d2',
        line: {
          color: '#0d47a1',
          width: 1
        }
      }
    };

    const layout: any = {
      title: 'Conflict Probability Distribution',
      xaxis: { title: 'Conflict Probability (%)' },
      yaxis: { title: 'Number of Countries' },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 50, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('probability-distribution', [trace], layout, config);
  }

  renderRiskLevelPie(): void {
    const riskCounts: any = {
      'LOW': 0,
      'MEDIUM': 0,
      'HIGH': 0,
      'CRITICAL': 0
    };

    this.filteredPredictions.forEach(p => {
      riskCounts[p.risk_level]++;
    });

    const trace: any = {
      labels: Object.keys(riskCounts),
      values: Object.values(riskCounts),
      type: 'pie',
      marker: {
        colors: ['#4caf50', '#ffeb3b', '#ff9800', '#f44336']
      }
    };

    const layout: any = {
      title: 'Countries by Risk Level',
      paper_bgcolor: '#fafafa',
      margin: { t: 40, b: 20, l: 20, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('risk-level-pie', [trace], layout, config);
  }

  renderTopCountriesBar(): void {
    const top10 = [...this.filteredPredictions]
      .sort((a, b) => b.conflict_probability - a.conflict_probability)
      .slice(0, 10);

    const trace: any = {
      x: top10.map(p => p.conflict_probability * 100),
      y: top10.map(p => p.country),
      type: 'bar',
      orientation: 'h',
      marker: {
        color: top10.map(p => {
          if (p.risk_level === 'CRITICAL') return '#f44336';
          if (p.risk_level === 'HIGH') return '#ff9800';
          if (p.risk_level === 'MEDIUM') return '#ffeb3b';
          return '#4caf50';
        })
      }
    };

    const layout: any = {
      title: 'Top 10 Countries by Conflict Probability',
      xaxis: { title: 'Conflict Probability (%)' },
      yaxis: { automargin: true },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 120, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('top-countries-bar', [trace], layout, config);
  }

  renderPerformanceChart(): void {
    if (!this.performance) return;

    const metrics = this.performance.best_model_metrics;
    const metricNames = ['ROC-AUC', 'F1 Score', 'Precision', 'Recall', 'Accuracy'];
    const metricValues = [
      metrics.roc_auc * 100,
      metrics.f1_score * 100,
      metrics.precision * 100,
      metrics.recall * 100,
      metrics.accuracy * 100
    ];

    const trace: any = {
      x: metricNames,
      y: metricValues,
      type: 'bar',
      marker: {
        color: '#1976d2'
      },
      text: metricValues.map(v => v.toFixed(1) + '%'),
      textposition: 'auto'
    };

    const layout: any = {
      title: `Model Performance (${this.performance.best_model})`,
      yaxis: { title: 'Score (%)', range: [0, 100] },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 50, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('performance-chart', [trace], layout, config);
  }

  renderRiskScoresChart(): void {
    if (this.riskScores.length === 0) return;

    const trace: any = {
      x: this.riskScores.map(s => s.country),
      y: this.riskScores.map(s => s.risk_score),
      type: 'bar',
      marker: {
        color: this.riskScores.map(s => {
          if (s.risk_level === 'CRITICAL') return '#f44336';
          if (s.risk_level === 'HIGH') return '#ff9800';
          if (s.risk_level === 'MEDIUM') return '#ffeb3b';
          return '#4caf50';
        })
      },
      text: this.riskScores.map(s => s.risk_score.toFixed(1)),
      textposition: 'auto'
    };

    const layout: any = {
      title: 'Top 10 Risk Countries',
      yaxis: { title: 'Risk Score (0-100)' },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 80, l: 50, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('risk-scores-chart', [trace], layout, config);
  }

  getRiskLevelColor(level: string): string {
    const colors: any = {
      'CRITICAL': '#f44336',
      'HIGH': '#ff9800',
      'MEDIUM': '#ffeb3b',
      'LOW': '#4caf50'
    };
    return colors[level] || '#9e9e9e';
  }

  getTrendIcon(pred: ConflictPrediction): string {
    // This would need historical data - placeholder
    return '';
  }

  formatProbability(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }

  formatConfidence(conf: number): string {
    return `${(conf * 100).toFixed(1)}%`;
  }

  getTargetDateDisplay(horizon: number): string {
    const today = new Date();
    const targetDate = new Date(today);
    targetDate.setDate(today.getDate() + horizon);
    return targetDate.toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  }

  trackByStoryId(index: number, story: any): number {
    return story.id;
  }

  // Make Math available in template
  Math = Math;
}
