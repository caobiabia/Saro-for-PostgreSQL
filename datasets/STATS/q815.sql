select  count(*) from comments as c,          postHistory as ph,  		badges as b,          users as u  where u.Id = c.UserId 	and u.Id = ph.UserId 	and u.Id = b.UserId  AND b.Date>='2010-07-22 14:58:19'::timestamp  AND b.Date<='2014-09-13 22:15:14'::timestamp  AND c.Score=0  AND u.Views=1  AND u.DownVotes>=0  AND u.CreationDate>='2010-07-26 21:46:11'::timestamp  AND u.CreationDate<='2014-09-11 03:57:26'::timestamp;