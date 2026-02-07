import { Component, OnInit, AfterViewInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatChipsModule } from '@angular/material/chips';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';
import { MatListModule } from '@angular/material/list';
import { MatTabsModule } from '@angular/material/tabs';
import { ApiService } from '../../services/api.service';
import { PredictionResponse, NewsStory, ConflictPrediction } from '../../models/interfaces';
import * as Plotly from 'plotly.js-geo-dist';

@Component({
  selector: 'app-country-detail',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatChipsModule,
    MatProgressSpinnerModule,
    MatButtonModule,
    MatIconModule,
    MatListModule,
    MatTabsModule
  ],
  templateUrl: './country-detail.component.html',
  styleUrls: ['./country-detail.component.scss']
})
export class CountryDetailComponent implements OnInit, AfterViewInit {
  countryName: string = '';
  detail: PredictionResponse | null = null;
  recentNews: NewsStory[] = [];
  allHorizonPredictions: ConflictPrediction[] = [];
  loading = true;
  error: string | null = null;

  constructor(
    private route: ActivatedRoute,
    private apiService: ApiService
  ) {}

  ngOnInit(): void {
    this.route.params.subscribe(params => {
      this.countryName = params['name'];
      this.loadCountryDetail();
    });
  }

  ngAfterViewInit(): void {
    // Charts will be rendered after data loads
  }

  loadCountryDetail(): void {
    this.loading = true;
    this.error = null;

    this.apiService.getCountryPrediction(this.countryName).subscribe({
      next: (data) => {
        this.detail = data;
        this.allHorizonPredictions = data.predictions || [];
        this.recentNews = data.recent_news || [];
        this.loading = false;
        setTimeout(() => this.renderCharts(), 100);
      },
      error: (err) => {
        this.error = `Failed to load data for ${this.countryName}`;
        this.loading = false;
        console.error('Error loading country detail:', err);
      }
    });
  }

  renderCharts(): void {
    if (!this.detail) return;
    this.render7DayTrend();
    this.renderFeatureImportance();
    this.renderGaugeChart();
    this.renderHorizonComparison();
  }

  renderHorizonComparison(): void {
    if (!this.allHorizonPredictions || this.allHorizonPredictions.length === 0) return;

    const horizons = this.allHorizonPredictions.map(p => `${p.horizon_days} Days`);
    const probabilities = this.allHorizonPredictions.map(p => p.conflict_probability * 100);
    const colors = probabilities.map(p => {
      if (p >= 80) return '#f44336';
      if (p >= 60) return '#ff9800';
      if (p >= 40) return '#ffeb3b';
      return '#4caf50';
    });

    const trace: any = {
      x: horizons,
      y: probabilities,
      type: 'bar',
      marker: {
        color: colors,
        line: { color: '#333', width: 1 }
      },
      text: probabilities.map(p => `${p.toFixed(1)}%`),
      textposition: 'auto'
    };

    const layout: any = {
      title: 'Predictions by Forecast Horizon',
      xaxis: { title: 'Forecast Horizon' },
      yaxis: { title: 'Conflict Probability (%)', range: [0, 100] },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 50, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('horizon-chart', [trace], layout, config);
  }

  render7DayTrend(): void {
    if (!this.detail) return;

    const days = this.detail.trend_7day.map((_, i) => `Day ${i + 1}`);
    const probabilities = this.detail.trend_7day.map(v => v * 100);

    const trace: any = {
      x: days,
      y: probabilities,
      type: 'scatter',
      mode: 'lines+markers',
      line: {
        color: '#1976d2',
        width: 3
      },
      marker: {
        size: 8,
        color: probabilities.map(p => {
          if (p >= 80) return '#f44336';
          if (p >= 60) return '#ff9800';
          if (p >= 40) return '#ffeb3b';
          return '#4caf50';
        })
      },
      fill: 'tozeroy',
      fillcolor: 'rgba(25, 118, 210, 0.1)'
    };

    const layout: any = {
      title: '7-Day Conflict Probability Trend',
      xaxis: { title: 'Day' },
      yaxis: { title: 'Conflict Probability (%)', range: [0, 100] },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 50, r: 20 },
      shapes: [
        // Horizontal lines for risk levels
        {
          type: 'line',
          x0: 0,
          x1: 1,
          xref: 'paper',
          y0: 40,
          y1: 40,
          line: { color: '#ffeb3b', width: 1, dash: 'dash' }
        },
        {
          type: 'line',
          x0: 0,
          x1: 1,
          xref: 'paper',
          y0: 60,
          y1: 60,
          line: { color: '#ff9800', width: 1, dash: 'dash' }
        },
        {
          type: 'line',
          x0: 0,
          x1: 1,
          xref: 'paper',
          y0: 80,
          y1: 80,
          line: { color: '#f44336', width: 1, dash: 'dash' }
        }
      ]
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('trend-chart', [trace], layout, config);
  }

  renderFeatureImportance(): void {
    if (!this.detail) return;

    const features = this.detail.top_features;
    const featureNames = features.map(f => f.feature_name);
    const contributions = features.map(f => f.contribution);

    const trace: any = {
      x: contributions,
      y: featureNames,
      type: 'bar',
      orientation: 'h',
      marker: {
        color: '#1976d2',
        line: {
          color: '#0d47a1',
          width: 1
        }
      },
      text: contributions.map(c => c.toFixed(3)),
      textposition: 'auto'
    };

    const layout: any = {
      title: 'Top Contributing Features',
      xaxis: { title: 'Feature Contribution' },
      yaxis: { automargin: true },
      paper_bgcolor: '#fafafa',
      plot_bgcolor: '#ffffff',
      margin: { t: 40, b: 40, l: 200, r: 20 }
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('feature-importance-chart', [trace], layout, config);
  }

  renderGaugeChart(): void {
    if (!this.detail) return;

    const probability = this.detail.current_prediction.conflict_probability * 100;

    const trace: any = {
      type: 'indicator',
      mode: 'gauge+number+delta',
      value: probability,
      delta: { reference: 50 },
      gauge: {
        axis: { range: [0, 100] },
        bar: { color: '#1976d2' },
        steps: [
          { range: [0, 40], color: '#c8e6c9' },
          { range: [40, 60], color: '#fff9c4' },
          { range: [60, 80], color: '#ffe0b2' },
          { range: [80, 100], color: '#ffcdd2' }
        ],
        threshold: {
          line: { color: '#f44336', width: 4 },
          thickness: 0.75,
          value: 80
        }
      }
    };

    const layout: any = {
      title: 'Current Conflict Probability',
      paper_bgcolor: '#fafafa',
      margin: { t: 60, b: 20, l: 20, r: 20 },
      height: 300
    };

    const config: any = { responsive: true, displayModeBar: false };

    Plotly.newPlot('gauge-chart', [trace], layout, config);
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

  formatProbability(prob: number): string {
    return `${(prob * 100).toFixed(1)}%`;
  }

  formatConfidence(conf: number): string {
    return `${(conf * 100).toFixed(1)}%`;
  }
}
