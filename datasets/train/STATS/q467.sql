select  count(*) from postHistory as ph,          votes as v,  		users as u,  		badges as b  where u.Id = ph.UserId 	and u.Id = v.UserId 	and u.Id = b.UserId  AND ph.PostHistoryTypeId=36  AND u.DownVotes>=0  AND u.UpVotes>=0  AND u.UpVotes<=228  AND v.CreationDate>='2010-07-20 00:00:00'::timestamp;