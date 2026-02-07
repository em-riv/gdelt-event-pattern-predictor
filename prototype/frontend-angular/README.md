# Frontend - Angular Dashboard

Interactive web dashboard for GDELT Conflict Predictor using Angular 17 and Plotly visualizations.

## Quick Start

```bash
npm install
npm start
```

Dashboard opens at http://localhost:4200

## Features

### Interactive Dashboard
- Probability distribution histogram
- Risk level pie chart showing country distribution
- Top 10 countries bar chart
- Model performance metrics chart
- Interactive data table with country selection

### Advanced Filters
- Risk level filter (LOW, MEDIUM, HIGH, CRITICAL)
- Country multi-select dropdown
- Probability threshold slider (0-100%)
- Real-time filter application

### Country Detail Pages
- 7-day trend chart with risk level indicators
- Feature importance bar chart
- Gauge chart for current probability
- Detailed statistics and interpretation

## Prerequisites

- Node.js 18+
- Backend running at http://localhost:8000

## Architecture

### Main Components

- app.component.ts - Root component and navigation
- app.routes.ts - Route configuration
- services/ - API communication
- components/ - Feature components
- models/ - TypeScript interfaces

### Services

**api.service.ts**
- Fetches predictions from backend API
- Handles HTTP errors
- Caches results where applicable

## Build

### Development Build

```bash
npm run build
```

### Production Build

```bash
npm run build -- --configuration production
```

Output in dist/gdelt-conflict-predictor/

## Available Filters

### Risk Level Filter
Select one or more levels:
- LOW (0-40% probability)
- MEDIUM (40-60% probability)
- HIGH (60-80% probability)
- CRITICAL (80-100% probability)

### Country Filter
Multi-select dropdown with all available countries

### Probability Threshold
Slider from 0-100%:
- Shows only countries above threshold
- Updates charts in real-time

## Visualizations

All charts built with Plotly.js for interactivity:

### Dashboard Charts

1. Probability Distribution - Histogram across all countries
2. Risk Level Pie Chart - Breakdown by category (color-coded)
3. Top 10 Countries Bar Chart - Sorted by score
4. Model Performance - ROC-AUC, F1, Precision, Recall, Accuracy
5. Risk Scores Chart - Top 10 high-risk countries

### Country Detail Charts

1. 7-Day Trend Line - Risk trends over time
2. Feature Importance Bar - Top contributing features
3. Gauge Chart - Current probability visualization

## Configuration

### Backend URL

Edit src/app/app.config.ts:

```typescript
export const API_BASE_URL = 'http://localhost:8000';
```

For production, update to your deployed backend URL.

### Plotly Configuration

Map and chart settings in individual components.

## Development Server

```bash
ng serve
```

Navigate to http://localhost:4200/

## Code Generation

Generate new component:
```bash
ng generate component component-name
```

Generate service:
```bash
ng generate service service-name
```

## Dependencies

- Angular 17.0.0 - Framework
- Plotly.js 2.27.1 - Interactive charts
- Plotly Geo 3.3.1 - World map visualization
- RxJS 7.8.0 - Reactive programming
- Angular Material 17.0.0 - UI components

## Testing

```bash
ng test
```

## Linting

```bash
ng lint
```

## Deployment

### Static Hosting (Netlify, Vercel, AWS S3)

1. Build for production:
```bash
npm run build -- --configuration production
```

2. Deploy dist/ folder to static host

3. Configure backend URL environment variable

### Docker

Create Dockerfile:
```dockerfile
FROM node:18-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build -- --configuration production

FROM nginx:latest
COPY --from=build /app/dist/gdelt-conflict-predictor /usr/share/nginx/html
EXPOSE 80
```

Build and run:
```bash
docker build -t gdelt-frontend .
docker run -p 80:80 gdelt-frontend
```

## Environment Configuration

Development (src/environments/environment.ts):
```typescript
export const environment = {
  production: false,
  apiUrl: 'http://localhost:8000'
};
```

Production (src/environments/environment.prod.ts):
```typescript
export const environment = {
  production: true,
  apiUrl: 'https://api.example.com'
};
```

## API Endpoints

The frontend calls these backend endpoints:

- GET /predictions/{country} - Country prediction
- GET /predictions/top - Top risk countries
- GET /health - API health check
- POST /predictions/batch - Multiple countries

See backend README for full API documentation.

## Performance

### Bundle Size

- Original: ~2.5MB
- Minified: ~600KB
- Gzipped: ~180KB

### Optimization

```bash
npm run build -- --configuration production --stats-json
```

## Troubleshooting

### Cannot connect to API

Make sure backend is running:
```bash
cd prototype/backend
python main_api.py
```

### Module not found errors

Install dependencies:
```bash
npm install
```

### Charts not rendering

1. Check browser console for errors
2. Verify Plotly.js installed: `npm install plotly.js-dist-min`
3. Clear cache and reload
4. For maps: requires internet (Plotly CDN)

### CORS errors

Backend CORS configured for localhost:4200.
If using different port, update backend main_api.py.

## Browser Support

- Chrome/Edge 90+
- Firefox 88+
- Safari 14+

## Tech Stack

- Angular 17 (standalone components)
- Angular Material (UI components)
- Plotly.js (interactive visualizations)
- TypeScript (type safety)
- SCSS (styling)

## See Also

- [Backend README](../backend/README.md) for API details
- Main project [README](../../README.md) for architecture overview
- Angular docs: https://angular.io/docs
- Plotly docs: https://plotly.com/javascript/
