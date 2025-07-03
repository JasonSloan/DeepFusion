```bash
# 查询encoder decoder等组件的使用率

nvidia-smi dmon -h

    GPU statistics are displayed in scrolling format with one line
    per sampling interval. Metrics to be monitored can be adjusted
    based on the width of terminal window. Monitoring is limited to
    a maximum of 16 devices. If no devices are specified, then up to
    first 16 supported devices under natural enumeration (starting
    with GPU index 0) are used for monitoring purpose.
    It is supported on Tesla, GRID, Quadro and limited GeForce products
    for Kepler or newer GPUs under x64 and ppc64 bare metal Linux.
    Note: On MIG-enabled GPUs, querying the utilization of encoder,
    decoder, jpeg, ofa, gpu, and memory is not currently supported.

    Usage: nvidia-smi dmon [options]

    Options include:
    [-i | --id]:          Comma separated Enumeration index, PCI bus ID or UUID
    [-d | --delay]:       Collection delay/interval in seconds [default=1sec]
    [-c | --count]:       Collect specified number of samples and exit
    [-s | --select]:      One or more metrics [default=puc]
                          Can be any of the following:
                              p - Power Usage and Temperature
                              u - Utilization
                              c - Proc and Mem Clocks
                              v - Power and Thermal Violations
                              m - FB, Bar1 and CC Protected Memory
                              e - ECC Errors and PCIe Replay errors
                              t - PCIe Rx and Tx Throughput
    [N/A | --gpm-metrics]: Comma-separated list of GPM metrics (no space in between) to watch
                           Available metrics:
                               Graphics Activity       = 1
                               SM Activity             = 2
                               SM Occupancy            = 3
                               Integer Activity        = 4
                               Tensor Activity         = 5
                               DFMA Tensor Activity    = 6
                               HMMA Tensor Activity    = 7
                               IMMA Tensor Activity    = 9
                               DRAM Activity           = 10
                               FP64 Activity           = 11
                               FP32 Activity           = 12
                               FP16 Activity           = 13
                               PCIe TX                 = 20
                               PCIe RX                 = 21
                               NVDEC 0-7 Activity      = 30-37
                               NVJPG 0-7 Activity      = 40-47
                               NVOFA 0 Activity        = 50
                               NVLink Total RX         = 60
                               NVLink Total TX         = 61
                               NVLink L0-17 RX         = 62,64,66,...,96
                               NVLink L0-17 TX         = 63,65,67,...,97

    [N/A | --gpm-options]: options of which level of GPM metrics to monitor:
                              d  - Display Device level GPM Metrics only
                              m  - Display MIG level GPM Metrics only
                              dm - Display both Device and MIG level GPM Metrics only
                              md - Display both Device and MIG level GPM Metrics only
    [-o | --options]:     One or more from the following:
                              D - Include Date (YYYYMMDD) in scrolling output
                              T - Include Time (HH:MM:SS) in scrolling output
    [-f | --filename]:    Log to a specified file, rather than to stdout
    [-h | --help]:        Display help information
    [N/A | --format]:     Output format specifiers:
                               csv - Format dmon output as a CSV
                               nounit - Remove units line from dmon output
                               noheader - Remove heading line from dmon output
```

```bash
# 例如
dmon -s puct --gpm-metrics 30,31,32,33,34,35,36,37
```

