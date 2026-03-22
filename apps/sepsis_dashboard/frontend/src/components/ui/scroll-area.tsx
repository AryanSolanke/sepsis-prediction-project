import * as React from "react";
import { cn } from "../../lib/utils";

function ScrollArea({
  className,
  children,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("relative overflow-hidden", className)}
      {...props}
    >
      <div className="h-full w-full overflow-auto scrollbar-thin">
        {children}
      </div>
    </div>
  );
}

export { ScrollArea };
