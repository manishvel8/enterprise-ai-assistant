import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { DocumentsPageComponent } from './documents-page/documents-page.component';

const routes: Routes = [
  {
    path: '',
    component: DocumentsPageComponent,
  },
];

@NgModule({
  imports: [RouterModule.forChild(routes)],
  exports: [RouterModule],
})
export class DocumentsRoutingModule {}
