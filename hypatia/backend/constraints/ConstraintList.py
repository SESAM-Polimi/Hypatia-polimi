from hypatia.backend.constraints.Balance import Balance
from hypatia.backend.constraints.BalanceUnMetDemand import BalanceUnMetDemand
from hypatia.backend.constraints.TradeBalance import TradeBalance
from hypatia.backend.constraints.ResourceTechAvailability import ResourceTechAvailability
from hypatia.backend.constraints.TotalCapacityRegional import TotalCapacityRegional
from hypatia.backend.constraints.NewCapacityRegional import NewCapacityRegional
from hypatia.backend.constraints.TechEfficency import TechEfficency
from hypatia.backend.constraints.AnnualProductionRegional import AnnualProductionRegional
from hypatia.backend.constraints.EmissionCapRegional import EmissionCapRegional
from hypatia.backend.constraints.EmissionCapGlobal import EmissionCapGlobal
from hypatia.backend.constraints.EmissionConsumedRegional import EmissionConsumedRegional
from hypatia.backend.constraints.StorageMaxMinChange import StorageMaxMinChange
from hypatia.backend.constraints.StorageMaxFlowInOut import StorageMaxFlowInOut
from hypatia.backend.constraints.LineTotalCapacity import LineTotalCapacity
from hypatia.backend.constraints.TotalCapacityGlobal import TotalCapacityGlobal
from hypatia.backend.constraints.NewCapacityGlobal import NewCapacityGlobal
from hypatia.backend.constraints.AnnualProductionGlobal import AnnualProductionGlobal
from hypatia.backend.constraints.LineAvailability import LineAvailability
from hypatia.backend.constraints.LineNewCapacity import LineNewCapacity
from hypatia.backend.constraints.ProductionRamping import ProductionRamping
from hypatia.backend.constraints.RenewableProductionRegional import RenewableProductionRegional
from hypatia.backend.constraints.RenewableProductionGlobal import RenewableProductionGlobal  
from hypatia.backend.constraints.LandUsageRegional import LandUsageRegional
from hypatia.backend.constraints.LandUsageGlobal import LandUsageGlobal
# from hypatia.backend.constraints.ElectrolysisConsumption import ElectrolysisConsumption

CONSTRAINTS = [
    Balance,
    BalanceUnMetDemand,
    TradeBalance,
    ResourceTechAvailability,
    TotalCapacityRegional,
    NewCapacityRegional,
    TechEfficency,
    AnnualProductionRegional,
    EmissionCapRegional,
    EmissionCapGlobal,
    EmissionConsumedRegional,
    StorageMaxMinChange,
    StorageMaxFlowInOut,
    LineTotalCapacity,
    TotalCapacityGlobal,
    NewCapacityGlobal,
    AnnualProductionGlobal,
    RenewableProductionGlobal,
    LineAvailability,
    LineNewCapacity,
    # ProductionRamping,
    RenewableProductionRegional,
    LandUsageRegional,
    LandUsageGlobal,
    # ElectrolysisConsumption
]


