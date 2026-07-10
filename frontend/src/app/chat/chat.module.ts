import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChatRoutingModule } from './chat-routing.module';
import { ChatPageComponent } from './chat-page/chat-page.component';

@NgModule({
  imports: [
    CommonModule,
    ChatRoutingModule,
    ChatPageComponent,       // standalone component — use imports, not declarations
  ],
})
export class ChatModule {}
