select  count(*) from postHistory as ph,          posts as p,  		users as u,  		badges as b  where u.Id = p.OwnerUserId 	and p.OwnerUserId = ph.UserId 	and ph.UserId = b.UserId  AND b.Date<='2014-08-26 18:09:18'::timestamp  AND ph.PostHistoryTypeId=2  AND ph.CreationDate>='2010-12-22 17:57:02'::timestamp  AND ph.CreationDate<='2014-06-09 23:58:54'::timestamp  AND p.PostTypeId=2  AND p.FavoriteCount<=15  AND p.CreationDate<='2014-08-31 23:45:20'::timestamp  AND u.Reputation<=116  AND u.DownVotes>=0  AND u.UpVotes>=0  AND u.UpVotes<=355  AND u.CreationDate<='2014-09-04 12:57:35'::timestamp;