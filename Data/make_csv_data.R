library(readr)
library(lubridate)
library(dplyr)

MOSB <-read_delim('./МосБиржа_1min_06082015_07082016.csv', ';')
MOSB$`<TICKER>` <- NULL
MOSB$`<PER>` <- NULL
MOSB$`<HIGH>` <- NULL
MOSB$`<LOW>` <- NULL
MOSB$`<CLOSE>` <- NULL
MOSB$`<OPENINT>` <- NULL

GAZP <-read_delim('./ГАЗПРОМ_ао_1min_06082015_07082016.csv', ';')
GAZP$`<TICKER>` <- NULL
GAZP$`<PER>` <- NULL
GAZP$`<HIGH>` <- NULL
GAZP$`<LOW>` <- NULL
GAZP$`<CLOSE>` <- NULL
GAZP$`<OPENINT>` <- NULL

SBER <-read_delim('./Сбербанк_1min_06082015_07082016.csv', ';')
SBER$`<TICKER>` <- NULL
SBER$`<PER>` <- NULL
SBER$`<HIGH>` <- NULL
SBER$`<LOW>` <- NULL
SBER$`<CLOSE>` <- NULL
SBER$`<OPENINT>` <- NULL

GAZP$datetime <- paste(GAZP$`<DATE>`, GAZP$`<TIME>`, sep="-")
SBER$datetime <- paste(SBER$`<DATE>`, SBER$`<TIME>`, sep="-")
MOSB$datetime <- paste(MOSB$`<DATE>`, MOSB$`<TIME>`, sep="-")

GAZP$`<DATE>` <- ymd(GAZP$`<DATE>`)
SBER$`<DATE>` <- ymd(SBER$`<DATE>`)
MOSB$`<DATE>` <- ymd(MOSB$`<DATE>`)

check <- identical(GAZP$`<DATE>`, SBER$`<DATE>`) & identical(GAZP$`<TIME>`, SBER$`<TIME>`) 
check <- check & identical(GAZP$`<DATE>`, MOSB$`<DATE>`) & identical(GAZP$`<TIME>`, MOSB$`<TIME>`)
print(check)
stopifnot(check)

D <- tbl_df(data.frame(date = GAZP$`<DATE>`, time = GAZP$`<TIME>`))
D$datetime <- GAZP$datetime
D$time <- hms(paste0(substr(D$time,1,2), ':', substr(D$time,3,4), ':', substr(D$time,5,6)))
D$yday <- as.integer(yday(D$date))
D$mday <- mday(D$date)
weekday <- factor(as.integer(wday(D$date)))
D <- cbind(D, model.matrix(~weekday -1))
D$date <- as.integer(D$date)
D$time <- as.integer(as.duration(D$time))

MOSB$`<DATE>` <- NULL
MOSB$`<TIME>` <- NULL
SBER$`<DATE>` <- NULL
SBER$`<TIME>` <- NULL
GAZP$`<DATE>` <- NULL
GAZP$`<TIME>` <- NULL

output <- inner_join(MOSB, GAZP, by = 'datetime') %>% inner_join(SBER, by = 'datetime') %>% inner_join(D, by = 'datetime')
output$datetime <- NULL
#D <- tbl_df(cbind(MOSB$`<OPEN>`, MOSB$`<VOL>`, GAZP$`<OPEN>`, GAZP$`<VOL>`, SBER$`<OPEN>`, SBER$`<VOL>`, D))

output <- filter(output, weekday7 == 0, weekday1 == 0, time >= 36000, time <= 67500)
output$weekday1 <- NULL
output$weekday7 <- NULL
write_csv(output, 'data.csv')
