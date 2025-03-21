# Scheduler
调度器下维护着 BlockSpaceManager。它负责管理BlockAllocator（实际参与分配物理块的类）。BlockAllocator又分成gpu和cpu两种类型，分别管理这两类设备上的物理块。

![image](./image/img_4.png)

