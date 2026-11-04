# TEACHING: we choose the architecture; training learns every filter.
class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks, incoming = [], 3
        for outgoing in (32, 64, 128, 256):
            blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(), nn.MaxPool2d(2)))
            incoming = outgoing
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(256, 2))
    def forward(self, x):
        return self.head(self.blocks(x))
example_model = SmallCNN().to(DEVICE).eval()
print(f"Parameters: {sum(p.numel() for p in example_model.parameters()):,}")
with torch.no_grad():
    batch = torch.zeros(2, 3, img_size, img_size, device=DEVICE)
    for number, block in enumerate(example_model.blocks, 1):
        batch = block(batch)
        print(f"Block {number}: {tuple(batch.shape)}")
del example_model, batch
