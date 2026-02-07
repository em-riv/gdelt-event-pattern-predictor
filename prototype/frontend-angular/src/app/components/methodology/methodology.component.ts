import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { MatCardModule } from '@angular/material/card';
import { MatIconModule } from '@angular/material/icon';
import { MatButtonModule } from '@angular/material/button';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatDividerModule } from '@angular/material/divider';

@Component({
  selector: 'app-methodology',
  standalone: true,
  imports: [
    CommonModule,
    RouterModule,
    MatCardModule,
    MatIconModule,
    MatButtonModule,
    MatExpansionModule,
    MatDividerModule
  ],
  templateUrl: './methodology.component.html',
  styleUrls: ['./methodology.component.scss']
})
export class MethodologyComponent {
  
  dataSources = [
    {
      name: 'GDELT Project',
      description: 'Global Database of Events, Language, and Tone - monitors world broadcast, print, and web news in over 100 languages',
      metrics: '98M+ events analyzed',
      icon: 'public'
    },
    {
      name: 'Event Codes (CAMEO)',
      description: 'Conflict and Mediation Event Observations taxonomy for categorizing international events',
      metrics: '20 root event categories',
      icon: 'category'
    },
    {
      name: 'Goldstein Scale',
      description: 'Measures theoretical potential impact of events on country stability (-10 to +10)',
      metrics: 'Continuous scale',
      icon: 'trending_up'
    }
  ];

  features = [
    {
      category: 'Event-Based Features',
      items: [
        'Event counts by category (verbal/material conflict)',
        'Goldstein scale aggregates (mean, min, max, std)',
        'Actor diversity metrics',
        'Event intensity and frequency patterns'
      ]
    },
    {
      category: 'Temporal Features',
      items: [
        '7-day rolling averages',
        '14-day trend indicators',
        '30-day baseline comparisons',
        'Week-over-week change rates'
      ]
    },
    {
      category: 'Geopolitical Features',
      items: [
        'Regional conflict spillover indices',
        'Neighboring country risk scores',
        'Historical conflict patterns',
        'Bilateral relationship indicators'
      ]
    },
    {
      category: 'Sentiment Features',
      items: [
        'Average tone of news coverage',
        'Sentiment volatility measures',
        'Positive/negative event ratios',
        'Media attention intensity'
      ]
    }
  ];

  modelDetails = {
    name: 'GRU-based Forecasting Model',
    architecture: 'Gated Recurrent Unit (GRU) neural network with attention mechanism',
    training: 'Trained on 2015-2023 historical data',
    validation: 'Walk-forward validation with 2023-2024 test period',
    horizons: [7, 14, 30],
    metrics: {
      rocAuc: 0.89,
      f1Score: 0.72,
      precision: 0.78,
      recall: 0.67
    }
  };

  riskLevels = [
    { level: 'LOW', range: '0-25%', color: '#4caf50', description: 'Minimal conflict indicators, stable conditions' },
    { level: 'MEDIUM', range: '25-50%', color: '#ffeb3b', description: 'Elevated tensions, monitoring recommended' },
    { level: 'HIGH', range: '50-75%', color: '#ff9800', description: 'Significant risk factors, active monitoring required' },
    { level: 'CRITICAL', range: '75-100%', color: '#f44336', description: 'Severe indicators, immediate attention needed' }
  ];

  limitations = [
    'Predictions are probabilistic estimates, not certainties',
    'Model performance varies by region and conflict type',
    'Data availability may be limited for some countries',
    'Sudden, unprecedented events may not be captured',
    'Historical patterns may not predict novel conflict dynamics'
  ];
}
