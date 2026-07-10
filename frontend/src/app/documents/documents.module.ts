import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DocumentsRoutingModule } from './documents-routing.module';
import { DocumentsPageComponent } from './documents-page/documents-page.component';

@NgModule({
  imports: [
    CommonModule,
    DocumentsRoutingModule,
    DocumentsPageComponent,  // standalone component — use imports, not declarations
  ],
})
export class DocumentsModule {}
