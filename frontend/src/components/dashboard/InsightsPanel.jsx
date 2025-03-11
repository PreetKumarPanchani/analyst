import Icon from '@/components/ui/Icon';
import { ChevronDown, ChevronRight, TrendingUp, TrendingDown } from 'lucide-react';

<div className="flex items-center space-x-1">
  <span className="text-sm font-medium">More Insights</span>
  <Icon size="sm">
    <ChevronDown />
  </Icon>
</div>

<div className="flex items-center justify-between">
  <h3 className="text-sm font-medium">View Details</h3>
  <Icon size="sm">
    <ChevronRight />
  </Icon>
</div>

<div className="flex items-center space-x-2">
  <Icon size="md" className={trend > 0 ? "text-green-500" : "text-red-500"}>
    {trend > 0 ? <TrendingUp /> : <TrendingDown />}
  </Icon>
  <span className="text-lg font-semibold">{value}</span>
</div> 