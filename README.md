<h1>Giới thiệu</h1>
<p>Knowledge Distillation, hay còn gọi là chắc lọc tri thức là một phương pháp tối ưu các model deep leaning. Phương pháp này hoạt động trên
việc chuyển giao kiến thức từ một model lớn đến một model nhỏ. Như vậy thì model nhỏ có thể sử dụng để triển khai trên các phần cứng yếu hơn.</p>

<h1>Cách thức hoạt động</h1>
<p>Chúng ta có hai model. Model lớn là teacher model, đã được huấn luyện sẵn và model nhỏ là student model chưa được huấn luyện. Giờ công việc
  của chúng ta chỉ đơn giản là sử dụng teacher model để đánh nhãn cho các data sẽ được đưa vào huấn luyện ở student model.</p>
<img src=https://github.com/loki-lab/knowledge_distillation_pytorch/assets/128866042/b1272ee9-2696-4f68-95aa-e644177e1e37>

<h1>Hàm mất mát</h1>
Để thực hiện Knowlegde Distillation, ta cần phải sử dụng hàm loss như dưới đây.
<img src=https://github.com/loki-lab/knowledge_distillation_pytorch/assets/128866042/59f5b302-653f-4d6f-a437-fc35777064aa>
