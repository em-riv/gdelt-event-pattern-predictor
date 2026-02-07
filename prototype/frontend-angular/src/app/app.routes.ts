import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./components/dashboard/dashboard.component').then(m => m.DashboardComponent)
  },
  {
    path: 'country/:name',
    loadComponent: () => import('./components/country-detail/country-detail.component').then(m => m.CountryDetailComponent)
  },
  {
    path: 'methodology',
    loadComponent: () => import('./components/methodology/methodology.component').then(m => m.MethodologyComponent)
  },
  {
    path: '**',
    redirectTo: ''
  }
];
